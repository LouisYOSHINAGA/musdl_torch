from tqdm import tqdm
import torch as t
import torch.nn as nn
import torch.optim as optim
from typing import Callable
from typedef import *
from data import DataLoader

CriterionFn: TypeAlias = Callable[[PianoRollBatchTensor, t.Tensor], t.Tensor] \
                       | Callable[[PianoRollBatchTensor, PianoRollBatchTensor], t.Tensor]


class Trainer:
    def __init__(self, model: nn.Module, opt: optim.Optimizer, epochs: int,
                 train_dataloader: DataLoader, criterion_loss: CriterionFn, criterion_acc: CriterionFn,
                 test_dataloader: DataLoader | None =None) -> None:
        self.model: nn.Module = model
        self.opt: optim.Optimizer = opt
        self.epochs: int = epochs
        self.train_dataloader: DataLoader = train_dataloader
        self.criterion_loss: CriterionFn = criterion_loss
        self.criterion_acc: CriterionFn = criterion_acc
        self.test_dataloader: DataLoader | None = test_dataloader

    def __call__(self) -> tuple[TrainMetricLog, TrainMetricLog, TrainMetricLog, TrainMetricLog]:
        train_losses: TrainMetricLog = []
        train_accs: TrainMetricLog = []
        test_losses: TrainMetricLog = []
        test_accs: TrainMetricLog = []

        for _ in tqdm(range(self.epochs), desc="training progress"):
            train_loss, train_acc = self.train()
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            test_loss, test_acc = self.test()
            test_losses.append(test_loss)
            test_accs.append(test_acc)
        return train_losses, train_accs, test_losses, test_accs

    def train_only_loop(self) -> tuple[TrainMetricLog, TrainMetricLog]:
        train_losses: TrainMetricLog = []
        train_accs: TrainMetricLog = []
        for _ in tqdm(range(self.epochs), desc="training progress (training only)"):
            train_loss, train_acc = self.train()
            train_losses.append(train_loss)
            train_accs.append(train_acc)
        return train_losses, train_accs

    def train(self) -> tuple[float, float]:
        self.model.train()
        epoch_total_loss: float = 0
        epoch_total_acc: float = 0
        for prbt, kmbt in self.train_dataloader:
            self.opt.zero_grad()
            pred_kmbt: t.Tensor = self.model(prbt)
            kmbt = kmbt.to(self.model.device)
            loss: t.Tensor = self.criterion_loss(pred_kmbt, kmbt)
            loss.backward()
            epoch_total_loss += loss.item()
            epoch_total_acc += self.criterion_acc(pred_kmbt, kmbt).item()
            self.opt.step()
        return epoch_total_loss / len(self.train_dataloader), epoch_total_acc / len(self.train_dataloader)

    def test(self) -> tuple[float, float]:
        assert self.test_dataloader is not None, f"Test dataloader is not given."
        self.model.eval()
        epoch_total_loss: float = 0
        epoch_total_acc: float = 0
        with t.no_grad():
            for prbt, kmbt in self.test_dataloader:
                pred_kmbt: t.Tensor = self.model(prbt)
                kmbt = kmbt.to(self.model.device)
                epoch_total_loss += self.criterion_loss(pred_kmbt, kmbt).item()
                epoch_total_acc += self.criterion_acc(pred_kmbt, kmbt).item()
        return epoch_total_loss / len(self.test_dataloader), epoch_total_acc / len(self.test_dataloader)