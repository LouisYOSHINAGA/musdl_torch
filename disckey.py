import fire
import torch as t
import torch.nn as nn
from torch.optim import Adam
from typing import TypeAlias, Any
from typedef import *
from hparam import HyperParams
from data import MIDIChoraleDataLoader
from train import Trainer
from util import setup, lossfn_binary_cross_entropy, accfn_binary_accuracy
from plot import plot_train_log


PianoRollBowBatchTensor: TypeAlias = t.Tensor  # (batch, keyclass)


class KeyDiscNet(nn.Module):
    def __init__(self, hps: HyperParams) -> None:
        super().__init__()
        self.device: str = hps.general_device

        self.kdnet: nn.Sequential = nn.Sequential()
        hidden_dims: list[int] = [N_KEY_CLASS] + hps.kdn_hidden_dims + [1]
        for i in range(1, len(hidden_dims)):
            self.kdnet.add_module(f"fc{i:2d}", nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.kdnet.add_module(f"sigmoid{i:2d}", nn.Sigmoid())

    def preprocess(self, prbt: PianoRollBatchTensor) -> PianoRollBowBatchTensor:
        assert len(prbt.shape) == 3  # (batch, time, note)
        prbowbt_bn: t.Tensor = t.sum(prbt, dim=1)  # (batch, note)
        prbowbt_bok: t.Tensor = prbowbt_bn.view(prbt.shape[0], -1, N_KEY_CLASS)  # (batch, 8ve, keyclass)
        prbowbt: PianoRollBowBatchTensor = t.sum(prbowbt_bok, dim=1)  # (batch, keyclass)
        assert prbowbt.shape == (prbt.shape[0], N_KEY_CLASS)
        return prbowbt

    def forward(self, prbt: PianoRollBatchTensor) -> t.Tensor:
        return self.kdnet(self.preprocess(prbt).to(self.device)).squeeze()  # (batch, )

    @t.no_grad()
    def inference(self, prbt: PianoRollBatchTensor) -> t.Tensor:
        kmbt: t.Tensor = self.forward(prbt)
        return (kmbt >= 0.5).int()


def print_compare(trainer: Trainer) -> None:
    assert isinstance(trainer.test_dataloader, MIDIChoraleDataLoader)
    trainer.test_dataloader.set_modes("f")

    label2keymode: list[str] = ["Major", "Minor"]
    n_data: int = 0
    n_true_pos: int = 0
    n_false_pos: int = 0
    n_false_neg: int = 0
    n_true_neg: int = 0

    trainer.logger(f"\n{'=' * 60}")
    for filenames, (prbt, tkmbt) in trainer.test_dataloader:
        pkmbt: t.Tensor = trainer.inference(prbt)
        tkmbt = tkmbt.int().squeeze()
        n_data += len(prbt)
        for filename, pkm, tkm in zip(filenames, pkmbt, tkmbt):
            trainer.logger((
                f"{filename:>15}: "
                f"target={label2keymode[tkm]}, predict={label2keymode[pkm]} "
                f"[{'correct' if pkm == tkm else 'incorrect'}]"
            ))
            if pkm == 0 and tkm == 0:
                n_true_pos += 1
            elif pkm == 0 and tkm == 1:
                n_false_pos += 1
            elif pkm == 1 and tkm == 0:
                n_false_neg += 1
            elif pkm == 1 and tkm == 1:
                n_true_neg += 1
    n_data: int = n_true_pos + n_false_pos + n_false_neg + n_true_neg

    trainer.logger(f"{'=' * 60}")
    trainer.logger(f"Accuracy : {(n_true_pos+n_true_neg)/n_data:.5f} (={n_true_pos+n_true_neg}/{n_data})")
    trainer.logger(f"Precition: {n_true_pos/(n_true_pos+n_false_pos):.5f} (={n_true_pos}/{n_true_pos+n_false_pos})")
    trainer.logger(f"Recall   : {n_true_pos/(n_true_pos+n_false_neg):.5f} (={n_true_pos}/{n_true_pos+n_false_neg})")


def run(**kwargs: Any) -> None:
    trainer: Trainer = setup(model_class=KeyDiscNet, opt_class=Adam,
                             loss=lossfn_binary_cross_entropy, acc=accfn_binary_accuracy,
                             **kwargs, data_is_sep_part=False, conf="!fk")
    train_losses, train_accs, test_losses, test_accs = trainer()
    plot_train_log(train_losses, train_accs, test_losses, test_accs, is_save=True, logger=trainer.logger)
    print_compare(trainer)

if __name__ == "__main__":
    fire.Fire(run)