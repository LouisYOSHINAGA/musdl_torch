import fire
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Any
from typedef import *
from hparam import HyperParams
from data import MIDIChoraleDataLoader
from train import Trainer
from util import setup, rnn_general, lossfn_cross_entropy, accfn_accuracy
from plot import plot_train_log, plot_pianorolls, save_midi, scatter


class Encoder(nn.Module):
    def __init__(self, hps: HyperParams) -> None:
        super().__init__()
        self.device: str = hps.general_device

        self.rnn: nn.RNNBase = rnn_general(
            rnn_type=hps.cmp_enc_rnn_type,
            input_size=hps.data_note_high-hps.data_note_low+1,  # [note_low, note_high) \cup {rest}
            hidden_size=hps.cmp_enc_rnn_hidden_size,
            num_layers=hps.cmp_enc_num_layers,
            bidirectional=hps.cmp_enc_bidirectional,
            dropout=hps.cmp_enc_dropout,
            batch_first=True,
            device=hps.general_device,
        )
        self.fc: nn.Module = nn.Linear(hps.cmp_enc_rnn_hidden_size, hps.cmp_hidden_size)

    def forward(self, prbt: PianoRollBatchTensor) -> LatentBatchTensor:
        ys, _ = self.rnn(prbt.to(self.device))  # (batch, time, dim), (layer, time, dim)
        return self.fc(ys[:, -1, :])  # (batch, dim)


class Decoder(nn.Module):
    def __init__(self, hps: HyperParams) -> None:
        super().__init__()
        self.sequence_length: int = hps.data_resolution_nth_note * hps.data_length_bars
        self.n_note_class: int = hps.data_note_high - hps.data_note_low + 1  # [note_low, note_high) \cup {rest}

        self.rnn: nn.RNNBase = rnn_general(
            rnn_type=hps.cmp_dec_rnn_type,
            input_size=hps.cmp_hidden_size,
            hidden_size=hps.cmp_dec_rnn_hidden_size,
            num_layers=hps.cmp_dec_num_layers,
            bidirectional=hps.cmp_dec_bidirectional,
            dropout=hps.cmp_dec_dropout,
            batch_first=True,
            device=hps.general_device,
        )
        self.fc: nn.Module = nn.Linear(hps.cmp_dec_rnn_hidden_size, self.n_note_class)

    def forward(self, xs: LatentBatchTensor) -> PianoRollBatchTensor:
        ys: t.Tensor = xs.unsqueeze(1).repeat(1, self.sequence_length, 1)  # (batch, 1, dim) -> (batch, time, dim)
        ys, _ = self.rnn(ys)  # (batch, time, dim), (layer, time, dim)
        return self.fc(ys).reshape(-1, self.n_note_class)  # (batch, time, note) -> (batch*time, note)

    @t.no_grad()
    def reconstruct(self, xs: LatentBatchTensor) -> PianoRollBatchTensor:
        ys: t.Tensor = xs.unsqueeze(1).repeat(1, self.sequence_length, 1)  # (batch, 1, dim) -> (batch, time, dim)
        ys, _ = self.rnn(ys)  # (batch, time, dim), (layer, time, dim)
        ys = self.fc(ys)  # (batch, time, note)
        return F.one_hot(t.argmax(ys, dim=-1), num_classes=self.n_note_class)  # (batch, time) -> (batch, time, note)


class AutoEncoder(nn.Module):
    def __init__(self, hps: HyperParams) -> None:
        super().__init__()
        self.device: str = hps.general_device
        self.enc: Encoder = Encoder(hps)
        self.dec: Decoder = Decoder(hps)

    def forward(self, prbt: PianoRollBatchTensor) -> PianoRollBatchTensor:
        return self.dec(self.enc(prbt))

    @t.no_grad()
    def reconstruct(self, prbt: PianoRollBatchTensor) -> PianoRollBatchTensor:
        return self.dec.reconstruct(self.enc(prbt))

    @t.no_grad()
    def compress(self, prbt: PianoRollBatchTensor) -> LatentBatchTensor:
        return self.enc(prbt)


def reconstruct(trainer: Trainer, title: str|None =None, index: int =0, is_train: bool =False,
              **plot_kwargs: Any) -> None:
    dataloader: DataLoader = trainer.train_dataloader if is_train else trainer.test_dataloader
    assert isinstance(dataloader, MIDIChoraleDataLoader)
    dataloader.set_modes("f!k")

    trainer.model.eval()
    fns, (xs, _) = next(iter(dataloader))
    ys: PianoRollBatchTensor = trainer.model.reconstruct(xs)
    x: PianoRollTensor = xs[index, :, :-1].to("cpu")  # get `index`-th data, remove rest
    y: PianoRollTensor = ys[index, :, :-1].to("cpu")  # get `index`-th data, remove rest

    trainer.logger(f"\nTarget MIDI file for inference: {fns[index]}")
    plot_pianorolls(x, y, n_bars=trainer.hps.data_length_bars,
                    note_low=trainer.hps.data_note_low, note_high=trainer.hps.data_note_high,
                    logger=trainer.logger, title=title, **plot_kwargs)
    save_midi([x, y], logger=trainer.logger, title=title, note_offset=trainer.hps.data_note_low)

def compress(trainer: Trainer, title: str|None =None, n_data: int =64, n_dim: int =3, is_train: bool =False,
             **plot_kwargs: Any) -> None:
    dataloader: DataLoader = trainer.train_dataloader if is_train else trainer.test_dataloader
    assert isinstance(dataloader, MIDIChoraleDataLoader)
    dataloader.set_modes(f"!fk")

    trainer.model.eval()
    maj_zs: LatentBatchTensor = t.empty(0, n_dim)
    min_zs: LatentBatchTensor = t.empty(0, n_dim)
    cur_n_data: int = 0
    for xs, _, ts in dataloader:
        zs: LatentBatchTensor = trainer.model.compress(xs).to("cpu")  # (batch, dim)
        ts = ts.squeeze()  # (key, 1) -> (key, )
        maj_zs = t.vstack([maj_zs, zs[ts == KEY_MAJOR, :n_dim]])
        min_zs = t.vstack([min_zs, zs[ts == KEY_MINOR, :n_dim]])
        cur_n_data += zs.shape[0]
        if n_data <= cur_n_data:
            break

    trainer.logger(f"\nVisualize latent space with {cur_n_data} data.")
    scatter([maj_zs, min_zs], ["major", "minor"], n_dim=n_dim, logger=trainer.logger, title=title, **plot_kwargs)


def run(**kwargs: Any) -> None:
    trainer: Trainer = setup(model_class=AutoEncoder, opt_class=Adam,
                             loss=lossfn_cross_entropy, acc=accfn_accuracy,
                             **kwargs, data_is_sep_part=True, data_is_recons=True)
    train_losses, train_accs, test_losses, test_accs = trainer()
    plot_train_log(train_losses, train_accs, test_losses, test_accs,
                   is_save=True, logger=trainer.logger)
    reconstruct(trainer, title="recons_train", is_train=True, is_save=True)
    reconstruct(trainer, title="recons_test", is_save=True)
    compress(trainer, title="latent_train", is_train=True, is_save=True)
    compress(trainer, title="latent_test", is_save=True)

if __name__ == "__main__":
    fire.Fire(run)