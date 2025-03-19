import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.functional import binary_accuracy, multiclass_accuracy
from torch.utils.data import DataLoader
from typing import Any
from typedef import *
from hparam import HyperParams, setup_hyperparams
from log import Logger, setup_logger
from data import setup_dataloaders, MIDIChoraleDataLoader
from train import Trainer


def setup(model_class: type[nn.Module], opt_class: type[Optimizer],
          loss: CriterionFn, acc: CriterionFn, conf: str ="", **hps_kwargs: Any) -> Trainer:
    hps: HyperParams = setup_hyperparams(**hps_kwargs)
    logger: Logger = setup_logger(hps)
    train_dataloader, test_dataloader = setup_dataloaders(hps, logger, conf=conf)

    model: nn.Module = model_class(hps).to(hps.general_device)
    opt: Optimizer = opt_class(model.parameters(), lr=hps.train_lr)

    trainer: Trainer = Trainer(model, opt, hps, logger,
                               train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                               criterion_loss=loss, criterion_acc=acc)
    return trainer


def rnn_general(rnn_type: str, **rnn_kwargs: Any) -> nn.RNNBase:
    rnns: dict[str, type[nn.RNNBase]] = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
    assert rnn_type in rnns.keys(), f"Unexpected RNN type '{rnn_type}'. Available RNN type: {list(rnns.keys())}."
    return rnns[rnn_type](**rnn_kwargs)


def lossfn_binary_cross_entropy(input: t.Tensor, target: t.Tensor) -> t.Tensor:
    return F.binary_cross_entropy(input, target.squeeze())

def lossfn_cross_entropy(input: t.Tensor, target: t.Tensor) -> t.Tensor:
    return F.cross_entropy(input, target.reshape(-1))

def lossfn_elbo(inputs: tuple[t.Tensor, t.Tensor], target: t.Tensor) -> t.Tensor:
    recons, kl_loss = inputs
    recons_loss: t.Tensor = F.cross_entropy(recons, target.reshape(-1))  # - E[ log(p(x|z)) ]
    return recons_loss + kl_loss

def lossfn_binary_elbo(inputs: tuple[t.Tensor, t.Tensor], target: t.Tensor) -> t.Tensor:
    recons, kl_loss = inputs
    recons_loss: t.Tensor = F.binary_cross_entropy(recons, target)
    return recons_loss + kl_loss

def accfn_binary_accuracy(input: t.Tensor, target: t.Tensor) -> t.Tensor:
    return binary_accuracy(input, target.squeeze())

def accfn_accuracy(input: t.Tensor, target: t.Tensor) -> t.Tensor:
    return multiclass_accuracy(input, target.reshape(-1))

def accfn_accuracy_for_elbo(inputs: tuple[t.Tensor, t.Tensor], target: t.Tensor) -> t.Tensor:
    return multiclass_accuracy(inputs[0], target.reshape(-1))

def accfn_binary_accuracy_elbo(inputs: tuple[t.Tensor, t.Tensor], target: t.Tensor) -> t.Tensor:
    return ((inputs[0] >= 0.5).float() == target).float().sum() / target.numel()


def get_midi_chorale_dataloader(trainer: Trainer, is_train: bool =False, mode: str ="") -> MIDIChoraleDataLoader:
    dataloader: DataLoader = trainer.train_dataloader if is_train else trainer.test_dataloader
    assert isinstance(dataloader, MIDIChoraleDataLoader)
    dataloader.set_modes(mode)
    return dataloader