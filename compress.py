import fire
import torch as t
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_accuracy
from typing import Any
from typedef import *
from hparam import HyperParams, setup_hyperparams
from data import setup_dataloaders
from train import Trainer
from util import plot_train_log, plot_save_midi


class Encoder(nn.Module):
    def __init__(self, hps: HyperParams) -> None:
        super().__init__()
        self.device: str = hps.general_device

        assert hps.cmp_enc_rnn_type in ["rnn", "lstm", "gru"], f"Unexpected RNN type '{hps.cmp_enc_rnn_type}'."
        self.rnn: nn.RNNBase = {
            'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU
        }[hps.hrm_rnn_type](
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
        ys = self.fc(ys[:, -1, :])  # (batch, 1, dim)
        return ys


class Decoder(nn.Module):
    def __init__(self, hps: HyperParams) -> None:
        super().__init__()

        self.sequence_length: int = hps.data_resolution_nth_note * hps.data_length_bars
        self.n_note_class: int = hps.data_note_high - hps.data_note_low + 1  # [note_low, note_high) \cup {rest}
        assert hps.cmp_enc_rnn_type in ["rnn", "lstm", "gru"], f"Unexpected RNN type '{hps.cmp_enc_rnn_type}'."
        self.rnn: nn.RNNBase = {
            'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU
        }[hps.hrm_rnn_type](
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
        ys: t.Tensor = xs.repeat(1, self.sequence_length, 1)  # (batch, time, dim)
        ys, _ = self.rnn(ys)  # (batch, time, dim), (layer, time, dim)
        return self.fc(ys).reshape(-1, self.n_note_class)  # (batch, time, note) -> (batch*time, note)

    def inference(self, xs: t.Tensor) -> NoteSequenceBatchTensor:
        ys: t.Tensor = xs.repeat(1, self.sequence_length, 1)  # (batch, time, dim)
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

    def inference(self, prbt: PianoRollBatchTensor) -> PianoRollBatchTensor:
        return self.dec.inference(self.enc(prbt))


def cross_entropy_for_sequence_classify(input: t.Tensor, target: t.Tensor) -> t.Tensor:
    return F.cross_entropy(input, target.reshape(-1))

def multiclass_accuracy_for_sequence_classify(input: t.Tensor, target: t.Tensor) -> t.Tensor:
    return multiclass_accuracy(input, target.reshape(-1))


def run(**kwargs: Any) -> None:
    hps: HyperParams = setup_hyperparams(**kwargs, data_is_sep_part=True)
    train_dataloader, test_dataloader = setup_dataloaders(hps)
    model: AutoEncoder =  AutoEncoder(hps).to(hps.general_device)
    opt: Adam = Adam(model.parameters(), lr=hps.train_lr)
    trainer: Trainer = Trainer(model, opt, hps,
                               train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                               criterion_loss=cross_entropy_for_sequence_classify,
                               criterion_acc=multiclass_accuracy_for_sequence_classify)
    train_losses, train_accs, test_losses, test_accs = trainer()
    plot_train_log(train_losses, train_accs, test_losses, test_accs)

    plot_save_midi(train_dataloader, trainer.inference, hps)
    plot_save_midi(test_dataloader, trainer.inference, hps)

if __name__ == "__main__":
    fire.Fire(run)