import os
from tqdm import tqdm
import torch as t
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Any
from typedef import *
from hparam import HyperParams
from data import DataLoader
from util import get_time

CriterionFn: TypeAlias = Callable[[t.Tensor, t.Tensor], t.Tensor] \
                       | Callable[[tuple[t.Tensor, t.Tensor], t.Tensor], t.Tensor]


class Trainer:
    def __init__(self, model: nn.Module, opt: optim.Optimizer, hps: HyperParams,
                 train_dataloader: DataLoader, criterion_loss: CriterionFn, criterion_acc: CriterionFn,
                 test_dataloader: DataLoader|None =None) -> None:
        self.time: str = get_time()
        self.verbose: bool = hps.train_verbose

        self.model: nn.Module = model
        self.opt: optim.Optimizer = opt
        self.start_epoch: int = 0
        self.epochs: int = hps.train_epochs

        self.train_dataloader: DataLoader = train_dataloader
        self.criterion_loss: CriterionFn = criterion_loss
        self.criterion_acc: CriterionFn = criterion_acc
        self.test_dataloader: DataLoader|None = test_dataloader

        self.train_losses: TrainMetricLog = []
        self.train_accs: TrainMetricLog = []
        self.test_losses: TrainMetricLog = []
        self.test_accs: TrainMetricLog = []

        self.load(hps.train_load_path)

        self.save_period: int = hps.train_save_period
        self.save_path: str = hps.train_save_path if hps.train_save_path is not None \
                              else f"{hps.general_output_path}/model_weights_{self.time}.pth.tar"

    def __call__(self) -> tuple[TrainMetricLog, TrainMetricLog, TrainMetricLog, TrainMetricLog]:
        for epoch in tqdm(range(self.start_epoch, self.start_epoch+self.epochs), desc="training progress"):
            self.train()
            self.test()
            self.save(epoch)
        self.save()
        return self.train_losses, self.train_accs, self.test_losses, self.test_accs

    def train_only_loop(self) -> tuple[TrainMetricLog, TrainMetricLog]:
        for epoch in tqdm(range(self.start_epoch, self.start_epoch+self.epochs),
                          desc="training progress (training only)"):
            self.train()
            self.save(epoch)
        self.save()
        return self.train_losses, self.train_accs

    def train(self) -> None:
        self.model.train()
        epoch_total_loss: float = 0
        epoch_total_acc: float = 0
        for xs, ts in self.train_dataloader:
            self.opt.zero_grad()
            ys = self.model(xs)
            ts = ts.to(self.model.device)
            loss: t.Tensor = self.criterion_loss(ys, ts)
            loss.backward()
            epoch_total_loss += loss.item()
            epoch_total_acc += self.criterion_acc(ys, ts).item()
            self.opt.step()
        self.train_losses.append(epoch_total_loss / len(self.train_dataloader))
        self.train_accs.append(epoch_total_acc / len(self.train_dataloader))

    def test(self) -> None:
        assert self.test_dataloader is not None, f"Test dataloader is not given."
        self.model.eval()
        epoch_total_loss: float = 0
        epoch_total_acc: float = 0
        with t.no_grad():
            for xs, ts in self.test_dataloader:
                ys = self.model(xs)
                ts = ts.to(self.model.device)
                epoch_total_loss += self.criterion_loss(ys, ts).item()
                epoch_total_acc += self.criterion_acc(ys, ts).item()
        self.test_losses.append(epoch_total_loss / len(self.test_dataloader))
        self.test_accs.append(epoch_total_acc / len(self.test_dataloader))

    def inference(self, xs: t.Tensor) -> t.Tensor:
        return self.model.inference(xs)

    def load(self, path: str|None) -> None:
        if path is None:
            self.vprint(f"The path for the model weights to be loaded is not specified.")
            self.vprint(f"Start training the model from scratch.")
            return

        assert os.path.isfile(path), f"The path for model weights '{path}' does not exist."
        checkpoint: dict[str, Any] = t.load(path, map_location=self.model.device)
        self.model.load_state_dict(checkpoint['model'])
        self.opt.load_state_dict(checkpoint['opt'])
        self.start_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_loss']
        self.train_acc = checkpoint['train_acc']
        self.test_losses = checkpoint['test_loss']
        self.test_acc = checkpoint['test_acc']
        self.vprint(f"The model weights are loaded from '{path}'.")

    def save(self, epoch: int|None =None) -> None:
        if epoch is not None and (epoch + 1) % self.save_period != 0:
            return

        state_dict: dict[str, Any] = {
            'model': self.model.to('cpu').state_dict(),
            'opt': self.opt.state_dict(),
            'epoch': self.epochs if epoch is None else epoch+1,
            'train_loss': self.train_losses,
            'train_acc': self.train_accs,
            'test_loss': self.test_losses,
            'test_acc': self.test_accs,
        }
        t.save(state_dict, self.save_path)
        self.vprint(f"The model weights are saved in '{self.save_path}'.")

    def vprint(self, msg: str) -> None:
        if self.verbose:
            print(f"{msg}")