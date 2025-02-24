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
from util import rnn_general, plot_train_log, plot_pianorolls, save_midi


class HarmonyRNN(nn.Module):
    def __init__(self, hps: HyperParams) -> None:
        super().__init__()
        self.device: str = hps.general_device
        self.n_note_class: int = hps.data_note_high - hps.data_note_low + 1  # [note_low, note_high) \cup {rest}

        self.rnn: nn.RNNBase = rnn_general(
            rnn_type=hps.hrm_rnn_type,
            input_size=self.n_note_class,
            hidden_size=hps.hrm_rnn_hidden_size,
            num_layers=hps.hrm_rnn_num_layers,
            bidirectional=hps.hrm_rnn_bidirectional,
            dropout=hps.hrm_rnn_dropout,
            batch_first=True,
            device=hps.general_device,
        )
        self.fc: nn.Module = nn.Linear(hps.hrm_rnn_hidden_size, self.n_note_class)

    def forward(self, prbt: PianoRollBatchTensor) -> t.Tensor:
        ys, _  = self.rnn(prbt.to(self.device))  # (batch, time, dim), (layer, time, dim)
        return self.fc(ys).reshape(-1, self.n_note_class)  # (batch, time, note) -> (batch*time, note)

    def inference(self, prbt: PianoRollBatchTensor) -> NoteSequenceBatchTensor:
        ys, _ = self.rnn(prbt.to(self.device))
        ys = self.fc(ys)  # (batch, time, note)
        return F.one_hot(t.argmax(ys, dim=-1), num_classes=self.n_note_class)  # (batch, time) -> (batch, time, note)


def cross_entropy_for_sequence_classify(input: t.Tensor, target: t.Tensor) -> t.Tensor:
    return F.cross_entropy(input, target.reshape(-1))

def multiclass_accuracy_for_sequence_classify(input: t.Tensor, target: t.Tensor) -> t.Tensor:
    return multiclass_accuracy(input, target.reshape(-1))


def run(**kwargs: Any) -> None:
    hps: HyperParams = setup_hyperparams(**kwargs, data_is_sep_part=True)
    train_dataloader, test_dataloader = setup_dataloaders(hps)
    model: HarmonyRNN =  HarmonyRNN(hps).to(hps.general_device)
    opt: Adam = Adam(model.parameters(), lr=hps.train_lr)
    trainer: Trainer = Trainer(model, opt, hps,
                               train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                               criterion_loss=cross_entropy_for_sequence_classify,
                               criterion_acc=multiclass_accuracy_for_sequence_classify)
    train_losses, train_accs, test_losses, test_accs = trainer()
    plot_train_log(train_losses, train_accs, test_losses, test_accs)

    xs, _ = next(iter(test_dataloader))
    ys: PianoRollBatchTensor = trainer.inference(xs)

    x: PianoRollTensor = xs[0, :, :-1]
    y: PianoRollTensor = ys[0, :, :-1]
    save_midi([x, y], dirname=hps.general_output_path, note_offset=hps.data_note_low)
    plot_pianorolls(x, y, n_bars=hps.data_length_bars,
                    note_low=hps.data_note_low, note_high=hps.data_note_high)

if __name__ == "__main__":
    fire.Fire(run)