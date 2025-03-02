import os
from tqdm import tqdm
import torch as t
import torch.nn as nn
import torch.optim as optim
from typing import Any
from typedef import *
from hparam import HyperParams
from log import Logger
from data import DataLoader

eps: float = 1e-5


class Trainer:
    def __init__(self, model: nn.Module, opt: optim.Optimizer, hps: HyperParams, logger: Logger,
                 train_dataloader: DataLoader, test_dataloader: DataLoader,
                 criterion_loss: CriterionFn, criterion_acc: CriterionFn) -> None:
        self.logger: Logger = logger

        self.model: nn.Module = model
        self.opt: optim.Optimizer = opt
        self.start_epoch: int = 0
        self.epochs: int = hps.train_epochs

        self.train_dataloader: DataLoader = train_dataloader
        self.criterion_loss: CriterionFn = criterion_loss
        self.criterion_acc: CriterionFn = criterion_acc
        self.test_dataloader: DataLoader = test_dataloader

        self.train_losses: TrainMetricLog = []
        self.train_accs: TrainMetricLog = []
        self.test_losses: TrainMetricLog = []
        self.test_accs: TrainMetricLog = []

        self.load(hps.train_load_path)

        self.save_period: int = hps.train_save_period
        self.save_path: str = hps.train_save_path if hps.train_save_path is not None \
                              else f"{self.logger.outdir}/model_weights_{self.logger.time}.pth.tar"

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
        epoch_total_n_cor: int = 0
        n_data: int = 0
        for xs, ts in self.train_dataloader:
            self.opt.zero_grad()
            ys = self.model(xs)
            ts = ts.to(self.model.device)
            loss: t.Tensor = self.criterion_loss(ys, ts)
            loss.backward()
            self.opt.step()
            epoch_total_loss += loss.item()
            epoch_total_n_cor += int(len(xs) * self.criterion_acc(ys, ts).item() + eps)
            n_data += len(xs)
        self.train_losses.append(epoch_total_loss/n_data)
        self.train_accs.append(epoch_total_n_cor/n_data)

    def test(self) -> None:
        self.model.eval()
        epoch_total_loss: float = 0
        epoch_total_n_cor: int = 0
        n_data: int = 0
        with t.no_grad():
            for xs, ts in self.test_dataloader:
                ys = self.model(xs)
                ts = ts.to(self.model.device)
                epoch_total_loss += self.criterion_loss(ys, ts).item()
                epoch_total_n_cor += int(len(xs) * self.criterion_acc(ys, ts).item() + eps)
                n_data += len(xs)
        self.test_losses.append(epoch_total_loss/n_data)
        self.test_accs.append(epoch_total_n_cor/n_data)

    def inference(self, xs: t.Tensor) -> t.Tensor:
        self.model.eval()
        return self.model.inference(xs)

    def load(self, path: str|None) -> None:
        if path is None:
            self.logger(f"The path for the model weights to be loaded is not specified.")
            self.logger(f"Start training the model from scratch.\n")
            return

        assert os.path.isfile(path), f"The path for model weights '{path}' does not exist."
        checkpoint: dict[str, Any] = t.load(path, map_location=self.model.device)
        self.model.load_state_dict(checkpoint['model'])
        self.opt.load_state_dict(checkpoint['opt'])
        self.start_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_loss']
        self.train_accs = checkpoint['train_acc']
        self.test_losses = checkpoint['test_loss']
        self.test_accs = checkpoint['test_acc']
        self.logger(f"The model weights are loaded from '{path}'.\n")

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
        self.logger(f"[epoch {state_dict['epoch']:04d}] The model weights are saved in '{self.save_path}'.")

    @property
    def outdir(self) -> str:
        return self.logger.outdir

    @property
    def time(self) -> str:
        return self.logger.time