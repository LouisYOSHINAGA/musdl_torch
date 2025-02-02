import fire
import torch as t
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torcheval.metrics.functional import binary_accuracy
from typing import TypeAlias, Any
from typedef import *
from hparam import HyperParams, setup_hyperparams
from data import DataLoader, setup_dataloaders

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


def train_test_loop(model: KeyDiscNet, opt: Adam, train_dataloader: DataLoader, test_dataloader: DataLoader, \
                    epochs: int) -> tuple[TrainMetricLog, TrainMetricLog]:
    train_losses: list[float] = []
    test_accs: list[float] = []
    for epoch in range(epochs):
        print(f"epoch {epoch:03d}: ", end="")
        train_loss: float = train(model, opt, train_dataloader)
        train_losses.append(train_loss)
        print(f"train loss = {train_loss:.5f}, ", end="")
        test_acc: float = test(model, test_dataloader)
        test_accs.append(test_acc)
        print(f"test accuracy = {test_acc:.5f}")
    return train_losses, test_accs

def train(model: KeyDiscNet, opt: Adam, dataloader: DataLoader) -> float:
    model.train()
    epoch_total_loss: float = 0
    for prbt, kmbt in dataloader:
        opt.zero_grad()
        loss: t.Tensor = F.binary_cross_entropy(model(prbt), kmbt.to(model.device))
        epoch_total_loss += loss.item()
        loss.backward()
        opt.step()
    return epoch_total_loss / len(dataloader)

def test(model: KeyDiscNet, dataloader: DataLoader) -> float:
    model.eval()
    epoch_total_acc: float = 0
    with t.no_grad():
        for prbt, kmbt in dataloader:
            model.zero_grad()
            pred_kmbt: t.Tensor = model(prbt).squeeze()
            epoch_total_acc += binary_accuracy(pred_kmbt, kmbt.to(model.device).squeeze()).item()
    return epoch_total_acc / len(dataloader)

def run(**kwargs: Any) -> None:
    hps: HyperParams = setup_hyperparams(**kwargs, data_is_sep_part=False, data_is_return_key_mode=True)
    train_dataloader, test_dataloader = setup_dataloaders(hps)
    model: KeyDiscNet = KeyDiscNet(hps).to(hps.general_device)
    opt: Adam = Adam(model.parameters(), lr=hps.train_lr)
    losses, accs = train_test_loop(model, opt, train_dataloader, test_dataloader, hps.train_epochs)

if __name__ == "__main__":
    fire.Fire(run)