import fire, os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Any
from typedef import *
from hparam import HyperParams
from train import Trainer
from compmph import compress
from morphmph import morph
from util import setup, lossfn_binary_elbo, accfn_binary_accuracy_elbo
from plot import plot_train_log, plot_batch_pianoroll, save_batch_midi


class ConvolutionalEncoder(nn.Module):
    def __init__(self, hps: HyperParams) -> None:
        super().__init__()
        self.model_size: str = hps.gen_vae_model_size
        self.hidden_dim: int = hps.gen_vae_enc_hidden_dim
        self.latent_dim: int = hps.gen_vae_latent_dim
        self.beta: float = hps.gen_vae_beta
        self.device: str = hps.general_device

        if self.model_size == "small":
            self.convs: nn.Sequential = nn.Sequential(
                nn.Conv2d(
                    in_channels=1, out_channels=self.hidden_dim,
                    kernel_size=(1, hps.data_note_high-hps.data_note_low), stride=(1, 1), padding=0
                ),  # (batch, 1, seq=16, note) -> (batch, dim, seq=16, 1)
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                    kernel_size=(4, 1), stride=(4, 1), padding=0
                ),  # (batch, dim, 16, 1) -> (batch, dim, 4, 1)
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                    kernel_size=(4, 1), stride=(4, 1), padding=0
                ),  # (batch, dim, 4, 1) -> (batch, dim, 1, 1)
                nn.ReLU()
            )
        else:
            assert False, f"Unexpected model size '{self.model_size}'."

        self.fc_mean: nn.Module = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_lnvar: nn.Module = nn.Linear(self.hidden_dim, self.latent_dim)  # ln(sigma^2) = 2 * ln(sigma)

    def forward(self, prbt: PianoRollBatchTensor) -> tuple[LatentBatchTensor, t.Tensor]:
        assert self.model_size == "small" and prbt.shape[1] == 16 \
            or self.model_size == "large" and prbt.shape[1] == 48

        ys: t.Tensor = prbt.unsqueeze(1).to(self.device)  # (batch, 1, seq, note)
        ys = self.convs(ys)  # (batch, dim, 1, 1)
        ys = ys.reshape(-1, self.hidden_dim)  # (batch, dim)

        mean: t.Tensor = self.fc_mean(ys)  # (batch, dim)
        lnvar: t.Tensor = self.fc_mean(ys)  # (batch, dim)
        return self.reparameterize(mean, lnvar), self.kl_loss(mean, lnvar)

    def reparameterize(self, mean: t.Tensor, lnvar: t.Tensor) -> LatentBatchTensor:
        return mean + t.exp(lnvar/2) * t.randn_like(lnvar)

    def kl_loss(self, mean: t.Tensor, lnvar: t.Tensor) -> t.Tensor:
        return self.beta * t.mean(-t.sum(1 + lnvar - mean**2 - t.exp(lnvar), dim=1))  # KL[ q(z|x) || p(z) ]


class ConvolutionalDecoder(nn.Module):
    def __init__(self, hps: HyperParams) -> None:
        super().__init__()
        self.model_size: str = hps.gen_vae_model_size
        self.latent_dim: int = hps.gen_vae_latent_dim
        self.hidden_dim: int = hps.gen_vae_dec_hidden_dim
        self.device: str = hps.general_device

        self.fc: nn.Module = nn.Linear(self.latent_dim, self.hidden_dim)

        if self.model_size == "small":
            self.convs: nn.Sequential = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                    kernel_size=(4, 1), stride=(4, 1), padding=0
                ),  # (batch, dim, 1, 1) -> (batch, dim, 4, 1)
                nn.ReLU(),
                nn.ConvTranspose2d(
                    in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                    kernel_size=(4, 1), stride=(4, 1), padding=0
                ),  # (batch, dim, 4, 1) -> (batch, dim, 16, 1)
                nn.ReLU(),
                nn.ConvTranspose2d(
                    in_channels=self.hidden_dim, out_channels=1,
                    kernel_size=(1, hps.data_note_high-hps.data_note_low), stride=(1, 1), padding=0
                ),  # (batch, dim, 16=seq, 1) -> (batch, 1, 16=seq, note)
                nn.ReLU()
            )
        else:
            assert False, f"Unexpected model size '{self.model_size}'."

    def forward(self, zs: LatentBatchTensor) -> PianoRollBatchTensor:
        ys: t.Tensor = self.fc(zs).reshape(-1, self.hidden_dim, 1, 1)  # (batch, dim) -> (batch, dim, 1, 1)
        ys = self.convs(ys)  # (batch, 1, seq, note)
        return F.sigmoid(ys.squeeze())  # (batch, seq, note)

    @t.no_grad()
    def generate(self, n_sample: int) -> PianoRollBatchTensor:
        zs: t.Tensor = t.randn(size=(n_sample, self.latent_dim), device=self.device)  # (batch, dim)
        return self.forward(zs)  # (batch, seq, note)

    @t.no_grad()
    def morph(self, zs: LatentBatchTensor, idxs: list[int], n_intp: int) -> PianoRollBatchTensor:
        assert len(zs) >= 2 and len(idxs) >= 2 and n_intp >= 3
        vs: LatentBatchTensor = t.empty(n_intp, *zs.shape[1:])
        for i in range(n_intp):
            vs[i] = i/(n_intp-1) * zs[idxs[0]] + (n_intp-1-i)/(n_intp-1) * zs[idxs[1]]
        return self.forward(vs)  # (n_intp, seq, note)


class ConvolutionalVariationalAutoEncoder(nn.Module):
    def __init__(self, hps: HyperParams) -> None:
        super().__init__()
        self.enc: ConvolutionalEncoder = ConvolutionalEncoder(hps)
        self.dec: ConvolutionalDecoder = ConvolutionalDecoder(hps)

    def forward(self, prbt: PianoRollBatchTensor) -> tuple[PianoRollBatchTensor, t.Tensor]:
        z, kl_loss = self.enc(prbt)
        return self.dec(z), kl_loss

    @t.no_grad()
    def morph(self, prbt: PianoRollBatchTensor, idxs: list[int], n_intp: int) -> PianoRollBatchTensor:
        return self.dec.morph(self.enc(prbt)[0], idxs, n_intp)

    @t.no_grad()
    def compress(self, prbt: PianoRollBatchTensor) -> t.Tensor:
        return self.enc(prbt)[0]

    @t.no_grad()
    def generate(self, n_sample: int) -> PianoRollBatchTensor:
        return self.dec.generate(n_sample)


def generate(trainer: Trainer, n_sample: int, title: str ="generate", **plot_kwargs: Any) -> None:
    os.mkdir(f"{trainer.outdir}/{title}")
    trainer.logger(f"\nOutput directory for generation '{trainer.outdir}/{title}' is newly made.")

    trainer.model.eval()
    ys: PianoRollBatchTensor = trainer.model.generate(n_sample)

    trainer.logger(f"{'\n' if title is None else ''}Generate {n_sample} samples.")
    plot_batch_pianoroll(ys, trainer=trainer, title=title, **plot_kwargs)
    save_batch_midi(ys, trainer=trainer, title=title)


def run(**kwargs: Any) -> None:
    trainer: Trainer = setup(model_class=ConvolutionalVariationalAutoEncoder, opt_class=Adam,
                             loss=lossfn_binary_elbo, acc=accfn_binary_accuracy_elbo,
                             **kwargs, data_is_sep_part=False)
    train_losses, train_accs, test_losses, test_accs = trainer()
    plot_train_log(train_losses, train_accs, test_losses, test_accs, is_save=True, logger=trainer.logger)
    generate(trainer, n_sample=10, title="gen_poly", is_save=True)
    compress(trainer, title="latent_train", is_train=True, is_save=True)
    compress(trainer, title="latent_test", is_save=True)
    morph(trainer, n_intp=10, title="morph_train", is_train=True, is_save=True)
    morph(trainer, n_intp=10, title="morph_test", is_save=True)

if __name__ == "__main__":
    fire.Fire(run)