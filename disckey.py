import fire
import torch as t
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torcheval.metrics.functional import binary_accuracy
from typing import TypeAlias, Any
from typedef import *
from hparam import HyperParams, setup_hyperparams
from data import setup_dataloaders
from train import Trainer
from util import plot_train_log

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
        return self.kdnet(self.preprocess(prbt).to(self.device))  # (batch, ismaj)


def run(**kwargs: Any) -> None:
    hps: HyperParams = setup_hyperparams(**kwargs, data_is_sep_part=False, data_is_return_key_mode=True)
    train_dataloader, test_dataloader = setup_dataloaders(hps)
    model: KeyDiscNet = KeyDiscNet(hps).to(hps.general_device)
    opt: Adam = Adam(model.parameters(), lr=hps.train_lr)
    trainer: Trainer = Trainer(model, opt, hps.train_epochs,
                               train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                               criterion_loss=F.binary_cross_entropy, criterion_acc=binary_accuracy)
    train_losses, train_accs, test_losses, test_accs = trainer()
    plot_train_log(train_losses, train_accs, test_losses, test_accs)

if __name__ == "__main__":
    fire.Fire(run)