import pretty_midi as pm
import soundfile as sf
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torcheval.metrics.functional import binary_accuracy, multiclass_accuracy
from typing import Any
from typedef import *
from hparam import HyperParams, setup_hyperparams
from log import Logger, setup_logger
from data import MIDIChoraleDataLoader, setup_dataloaders
from train import Trainer


def setup(model_class: type[nn.Module], opt_class: type[Optimizer],
          loss: CriterionFn, acc: CriterionFn, **hps_kwargs: Any) -> Trainer:
    hps: HyperParams = setup_hyperparams(**hps_kwargs)
    logger: Logger = setup_logger(hps)
    train_dataloader, test_dataloader = setup_dataloaders(hps, logger)

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

def accfn_binary_accuracy(input: t.Tensor, target: t.Tensor) -> t.Tensor:
    return binary_accuracy(input, target.squeeze())

def accfn_accuracy(input: t.Tensor, target: t.Tensor) -> t.Tensor:
    return multiclass_accuracy(input, target.reshape(-1))


def plot_train_log(train_losses: TrainMetricLog, train_accs: TrainMetricLog,
                   test_losses: TrainMetricLog, test_accs: TrainMetricLog,
                   figsize: tuple[float, float] =(14, 5),
                   is_save: bool =False, logger: Logger|None =None, title: str ="loss_acc",
                   is_show: bool =False) -> None:
    if not is_save and not is_show:
        return

    fig = plt.figure(figsize=figsize)
    axl = fig.add_subplot(1, 2, 1)
    axl.set_xlabel("epoch")
    axl.set_ylabel("loss")
    axl.plot(range(len(train_losses)), train_losses, label="train loss")
    axl.plot(range(len(test_losses)), test_losses, label="test loss")
    axl.legend()

    axr = fig.add_subplot(1, 2, 2)
    axr.set_xlabel("epoch")
    axr.set_ylabel("accuracy")
    axr.plot(range(len(train_accs)), train_accs, label="train acc")
    axr.plot(range(len(test_accs)), test_accs, label="test acc")
    axr.legend()
    plt.tight_layout()

    if is_save:
        assert logger is not None
        save_path: str = f"{logger.outdir}/{title}_{logger.time}.png"
        plt.savefig(save_path, dpi=320, bbox_inches="tight")
        logger(f"Figure of loss and accuracy is saved in '{save_path}'.")
    if is_show:
        plt.show()
    else:
        plt.close()

def plot_pianoroll(pr: PianoRoll|PianoRollTensor, n_bars: int, note_low: int, note_high: int,
                   figsize: tuple[float, float] =(7, 5),
                   is_save: bool =False, logger: Logger|None =None, title: str|None =None,
                   is_show: bool =False) -> None:
    if not is_save and not is_show:
        return

    _, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("bar")
    ax.set_ylabel("note")
    ax.imshow(pr.T, origin="lower", aspect="auto", extent=[0, n_bars, note_low, note_high])  # type: ignore
    plt.tight_layout()

    if is_save:
        assert logger is not None
        save_path: str = f"{logger.outdir}/{title if title is not None else 'pianoroll'}_{logger.time}.png"
        plt.savefig(save_path, dpi=320, bbox_inches="tight")
        logger(f"Figure of pianoroll is saved in '{save_path}'.")
    if is_show:
        plt.show()
    else:
        plt.close()

def plot_pianorolls(pr0: PianoRoll|PianoRollTensor, pr1: PianoRoll|PianoRollTensor, n_bars: int,
                    note_low: int, note_high: int, figsize: tuple[float, float] =(14, 5),
                    is_save: bool =False, logger: Logger|None =None, title: str|None =None,
                    is_show: bool =False) -> None:
    if not is_save and not is_show:
        return

    fig = plt.figure(figsize=figsize)
    axl = fig.add_subplot(1, 2, 1)
    axl.set_xlabel("bar")
    axl.set_ylabel("note")
    axl.imshow(pr0.T, origin="lower", aspect="auto", extent=[0, n_bars, note_low, note_high])  # type: ignore

    axr = fig.add_subplot(1, 2, 2)
    axr.set_xlabel("bar")
    axr.set_ylabel("note")
    axr.imshow(pr1.T, origin="lower", aspect="auto", extent=[0, n_bars, note_low, note_high])  # type: ignore
    plt.tight_layout()

    if is_save:
        assert logger is not None
        save_path: str = f"{logger.outdir}/{title if title is not None else 'pianorolls'}_{logger.time}.png"
        plt.savefig(save_path, dpi=320, bbox_inches="tight")
        logger(f"Figure of pianorolls is saved in '{save_path}'.")
    if is_show:
        plt.show()
    else:
        plt.close()

def save_midi(prl: list[PianoRoll|PianoRollTensor], logger: Logger, title: str|None =None,
              resolution: int =480, note_offset: int =0, sr: int =44100,
              midext: str =".mid", wavext: str =".wav") -> None:
    midi: pm.PrettyMIDI = pm.PrettyMIDI(resolution=resolution)
    for pr in prl:
        assert len(pr.shape) == 2  # (time, note)
        inst: pm.Instrument = pm.Instrument(program=1)
        for i in range(pr.shape[0]):
            for j in range(pr.shape[1]):
                if pr[i, j] > 0.5:
                    inst.notes.append(pm.Note(start=i/2, end=(i+1)/2, pitch=j+note_offset, velocity=100))
        midi.instruments.append(inst)

    save_path: str = f"{logger.outdir}/{title if title is not None else 'midi'}_{logger.time}"
    midi.write(f"{save_path}{midext}")
    logger(f"Midi data is saved in '{save_path}{midext}'.")
    sf.write(f"{save_path}{wavext}", midi.synthesize(fs=sr), samplerate=sr)
    logger(f"Rendered wave data is saved in '{save_path}{wavext}'.")

def plot_save_midi(trainer: Trainer, title: str, index: int =0, is_train: bool =False,
                   **plot_kwargs: Any) -> None:
    dataloader: DataLoader = trainer.train_dataloader if is_train else trainer.test_dataloader
    assert isinstance(dataloader, MIDIChoraleDataLoader)
    dataloader.inference()

    fns, (xs, _) = next(iter(dataloader))
    ys: PianoRollBatchTensor = trainer.inference(xs)
    x: PianoRollTensor = xs[index, :, :-1].to("cpu")  # get `index`-th data, remove rest
    y: PianoRollTensor = ys[index, :, :-1].to("cpu")  # get `index`-th data, remove rest
    trainer.logger(f"\nTarget MIDI file for inference: {fns[index]}")
    plot_pianorolls(x, y, n_bars=trainer.hps.data_length_bars,
                    note_low=trainer.hps.data_note_low, note_high=trainer.hps.data_note_high,
                    logger=trainer.logger, title=title, **plot_kwargs)
    save_midi([x, y], logger=trainer.logger, title=title, note_offset=trainer.hps.data_note_low)