import fire
import torch as t
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from typing import Any
from typedef import *
from hparam import HyperParams
from train import Trainer
from util import setup, rnn_general, lossfn_cross_entropy, accfn_accuracy, plot_save_midi
from plot import plot_train_log


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

    def forward(self, prbt: PianoRollBatchTensor) -> t.Tensor:
        ys, _ = self.rnn(prbt.to(self.device))  # (batch, time, dim), (layer, time, dim)
        ys = self.fc(ys[:, -1, :])  # (batch, dim)
        return ys


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

    def forward(self, xs: t.Tensor) -> PianoRollBatchTensor:
        ys: t.Tensor = xs.unsqueeze(1).repeat(1, self.sequence_length, 1)  # (batch, 1, dim) -> (batch, time, dim)
        ys, _ = self.rnn(ys)  # (batch, time, dim), (layer, time, dim)
        return self.fc(ys).reshape(-1, self.n_note_class)  # (batch, time, note) -> (batch*time, note)

    @t.no_grad()
    def inference(self, xs: t.Tensor) -> NoteSequenceBatchTensor:
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
    def inference(self, prbt: PianoRollBatchTensor) -> PianoRollBatchTensor:
        return self.dec.inference(self.enc(prbt))

    @t.no_grad()
    def compress(self, prbt: PianoRollBatchTensor) -> t.Tensor:
        return self.enc(prbt)


def run(**kwargs: Any) -> None:
    trainer: Trainer = setup(model_class=AutoEncoder, opt_class=Adam,
                             loss=lossfn_cross_entropy, acc=accfn_accuracy,
                             **kwargs, data_is_sep_part=True, data_is_recons=True)
    train_losses, train_accs, test_losses, test_accs = trainer()
    plot_train_log(train_losses, train_accs, test_losses, test_accs,
                   is_save=True, logger=trainer.logger, is_show=True)
    plot_save_midi(trainer, title="recons_train", is_train=True, is_save=True, is_show=True)
    plot_save_midi(trainer, title="recons_test", is_save=True, is_show=True)
    # visualize_latent_space() # TODO

if __name__ == "__main__":
    fire.Fire(run)