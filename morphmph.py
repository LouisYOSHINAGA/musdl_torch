import fire, os, random
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Any
from typedef import *
from hparam import HyperParams
from data import MIDIChoraleDataLoader
from train import Trainer
from compmph import compress
from util import setup, rnn_general, lossfn_elbo, accfn_accuracy_for_elbo, get_midi_chorale_dataloader
from plot import plot_train_log, plot_batch_pianoroll, save_batch_midi


class Encoder(nn.Module):
    def __init__(self, hps: HyperParams) -> None:
        super().__init__()
        self.device: str = hps.general_device
        self.beta: float = hps.mrp_beta

        self.rnn: nn.RNNBase = rnn_general(
            rnn_type=hps.mrp_enc_rnn_type,
            input_size=hps.data_note_high-hps.data_note_low+1,  # [note_low, note_high) \cup {rest}
            hidden_size=hps.mrp_enc_rnn_hidden_size,
            num_layers=hps.mrp_enc_num_layers,
            bidirectional=hps.mrp_enc_bidirectional,
            dropout=hps.mrp_enc_dropout,
            batch_first=True,
            device=hps.general_device,
        )
        self.fc_mean: nn.Module = nn.Linear(hps.mrp_enc_rnn_hidden_size, hps.mrp_hidden_size)
        self.fc_lnvar: nn.Module = nn.Linear(hps.mrp_enc_rnn_hidden_size, hps.mrp_hidden_size)  # ln(sigma^2) = 2 * ln(sigma)

    def forward(self, prbt: PianoRollBatchTensor) -> tuple[t.Tensor, t.Tensor]:
        ys, _ = self.rnn(prbt.to(self.device))  # (batch, time, dim), (layer, time, dim)
        mean: t.Tensor = self.fc_mean(ys[:, -1, :])  # (batch, dim)
        lnvar: t.Tensor = self.fc_lnvar(ys[:, -1, :])  # (batch, dim)
        return self.reparameterize(mean, lnvar), self.kl_loss(mean, lnvar)

    def reparameterize(self, mean: t.Tensor, lnvar: t.Tensor) -> t.Tensor:
        return mean + t.exp(lnvar/2) * t.randn_like(lnvar)

    def kl_loss(self, mean: t.Tensor, lnvar: t.Tensor) -> t.Tensor:
        return self.beta * t.mean(-t.sum(1 + lnvar - mean**2 - t.exp(lnvar), dim=1))  # KL[ q(z|x) || p(z) ]


class Decoder(nn.Module):
    def __init__(self, hps: HyperParams) -> None:
        super().__init__()
        self.sequence_length: int = hps.data_resolution_nth_note * hps.data_length_bars
        self.n_note_class: int = hps.data_note_high - hps.data_note_low + 1  # [note_low, note_high) \cup {rest}
        self.hidden_size: int = hps.mrp_hidden_size
        self.device: str = hps.general_device

        self.rnn: nn.RNNBase = rnn_general(
            rnn_type=hps.mrp_dec_rnn_type,
            input_size=hps.mrp_hidden_size,
            hidden_size=hps.mrp_dec_rnn_hidden_size,
            num_layers=hps.mrp_dec_num_layers,
            bidirectional=hps.mrp_dec_bidirectional,
            dropout=hps.mrp_dec_dropout,
            batch_first=True,
            device=hps.general_device,
        )
        self.fc: nn.Module = nn.Linear(hps.mrp_dec_rnn_hidden_size, self.n_note_class)

    def forward(self, zs: t.Tensor) -> PianoRollBatchTensor:
        ys: t.Tensor = zs.unsqueeze(1).repeat(1, self.sequence_length, 1)  # (batch, 1, dim) -> (batch, time, dim)
        ys, _ = self.rnn(ys)  # (batch, time, dim), (layer, time, dim)
        return self.fc(ys).reshape(-1, self.n_note_class)  # (batch, time, note) -> (batch*time, note)

    @t.no_grad()
    def morph(self, zs: LatentBatchTensor, idxs: list[int], n_intp: int) -> PianoRollBatchTensor:
        assert len(zs) >= 2 and len(idxs) >= 2 and n_intp >= 3
        vs: LatentBatchTensor = t.empty(n_intp, *zs.shape[1:])
        for i in range(n_intp):
            vs[i] = i/(n_intp-1) * zs[idxs[0]] + (n_intp-1-i)/(n_intp-1) * zs[idxs[1]]

        ys: t.Tensor = vs.unsqueeze(1).repeat(1, self.sequence_length, 1)  # (batch, 1, dim) -> (batch, time, dim)
        ys, _ = self.rnn(ys)  # (batch, time, dim), (layer, time, dim)
        ys = self.fc(ys)  # (batch, time, note)
        return F.one_hot(t.argmax(ys, dim=-1), num_classes=self.n_note_class)  # (batch, time) -> (batch, time, note)

    @t.no_grad()
    def generate(self, n_sample: int) -> NoteSequenceBatchTensor:
        zs: t.Tensor = t.randn(size=(n_sample, self.hidden_size), device=self.device)  # (bs, dim)
        ys: t.Tensor = zs.unsqueeze(1).repeat(1, self.sequence_length, 1)  # (batch, 1, dim) -> (batch, time, dim)
        ys, _ = self.rnn(ys)  # (batch, time, dim), (layer, time, dim)
        ys = self.fc(ys)  # (batch, time, note)
        return F.one_hot(t.argmax(ys, dim=-1), num_classes=self.n_note_class)  # (batch, time) -> (batch, time, note)


class VariationalAutoEncoder(nn.Module):
    def __init__(self, hps: HyperParams) -> None:
        super().__init__()
        self.device: str = hps.general_device
        self.enc: Encoder = Encoder(hps)
        self.dec: Decoder = Decoder(hps)

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


def morph(trainer: Trainer, n_intp: int, title: str ="morph", is_train: bool =False, **plot_kwargs: Any) -> None:
    os.mkdir(f"{trainer.outdir}/{title}")
    trainer.logger(f"\nOutput directory for morphing '{trainer.outdir}/{title}' is newly made.")

    trainer.model.eval()
    dataloader: MIDIChoraleDataLoader = get_midi_chorale_dataloader(trainer, is_train=is_train, mode="f!k")
    fns, (xs, _) = next(iter(dataloader))

    idxs: list[int] = random.sample(range(len(xs)), k=2)
    ys: PianoRollBatchTensor = trainer.model.morph(xs, idxs, n_intp).to("cpu")[:, :, :-1]  # remove rest

    trainer.logger(f"{'\n' if title is None else ''}Target MIDI files for morphing: {fns[idxs[0]]}, {fns[idxs[1]]}")
    plot_batch_pianoroll(ys, trainer=trainer, title=title, **plot_kwargs)
    save_batch_midi(ys, trainer=trainer, title=title)

def generate(trainer: Trainer, n_sample: int, title: str ="generate", **plot_kwargs: Any) -> None:
    os.mkdir(f"{trainer.outdir}/{title}")
    trainer.logger(f"\nOutput directory for generation '{trainer.outdir}/{title}' is newly made.")

    trainer.model.eval()
    ys: PianoRollBatchTensor = trainer.model.generate(n_sample)

    trainer.logger(f"{'\n' if title is None else ''}Generate {n_sample} samples.")
    plot_batch_pianoroll(ys, trainer=trainer, title=title, **plot_kwargs)
    save_batch_midi(ys, trainer=trainer, title=title)


def run(**kwargs: Any) -> None:
    trainer: Trainer = setup(model_class=VariationalAutoEncoder, opt_class=Adam,
                             loss=lossfn_elbo, acc=accfn_accuracy_for_elbo,
                             **kwargs, data_is_sep_part=True, data_is_recons=True)
    train_losses, train_accs, test_losses, test_accs = trainer()
    plot_train_log(train_losses, train_accs, test_losses, test_accs, is_save=True, logger=trainer.logger)
    morph(trainer, n_intp=10, title="morph_train", is_train=True, is_save=True)
    morph(trainer, n_intp=10, title="morph_test", is_save=True)
    compress(trainer, title="latent_train", is_train=True, is_save=True)
    compress(trainer, title="latent_test", is_save=True)
    generate(trainer, n_sample=10, title="gen_sop", is_save=True)

if __name__ == "__main__":
    fire.Fire(run)