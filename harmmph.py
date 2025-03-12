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
from plot import plot_train_log, plot_pianorolls, save_midi


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

    @t.no_grad()
    def harmonize(self, prbt: PianoRollBatchTensor) -> PianoRollBatchTensor:
        ys, _ = self.rnn(prbt.to(self.device))  # (batch, time, dim), (layer, time, dim)
        ys = self.fc(ys)  # (batch, time, note)
        return F.one_hot(t.argmax(ys, dim=-1), num_classes=self.n_note_class)  # (batch, time) -> (batch, time, note)


def harmonize(trainer: Trainer, is_train: bool =False, index: int =0, title: str|None =None,
              **plot_kwargs: Any) -> None:
    dataloader: DataLoader = trainer.train_dataloader if is_train else trainer.test_dataloader
    assert isinstance(dataloader, MIDIChoraleDataLoader)
    dataloader.set_modes("f!k")

    trainer.model.eval()
    fns, (xs, _) = next(iter(dataloader))
    ys: PianoRollBatchTensor = trainer.model.harmonize(xs)
    x: PianoRollTensor = xs[index, :, :-1].to("cpu")  # get `index`-th data, remove rest
    y: PianoRollTensor = ys[index, :, :-1].to("cpu")  # get `index`-th data, remove rest

    trainer.logger(f"\nTarget MIDI file for inference: {fns[index]}")
    plot_pianorolls(x, y, hps=trainer.hps, logger=trainer.logger, title=title, **plot_kwargs)
    save_midi([x, y], logger=trainer.logger, title=title, note_offset=trainer.hps.data_note_low)


def run(**kwargs: Any) -> None:
    trainer: Trainer = setup(model_class=HarmonyRNN, opt_class=Adam,
                             loss=lossfn_cross_entropy, acc=accfn_accuracy,
                             **kwargs, data_is_sep_part=True, data_is_recons=False)
    train_losses, train_accs, test_losses, test_accs = trainer()
    plot_train_log(train_losses, train_accs, test_losses, test_accs, is_save=True, logger=trainer.logger)
    harmonize(trainer, title="hrm_alt", is_save=True)

if __name__ == "__main__":
    fire.Fire(run)