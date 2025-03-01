import pretty_midi as pm
import soundfile as sf
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.functional import binary_accuracy
from torch.utils.data import DataLoader
from typing import Any, Callable
from typedef import *
from hparam import HyperParams
from train import Logger


def rnn_general(rnn_type: str, **rnn_kwargs: Any) -> nn.RNNBase:
    rnns: dict[str, type[nn.RNNBase]] = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
    assert rnn_type in rnns.keys(), f"Unexpected RNN type '{rnn_type}'. Available RNN type: {list(rnns.keys())}."
    return rnns[rnn_type](**rnn_kwargs)


def lossfn_binary_cross_entropy(input: t.Tensor, target: t.Tensor) -> t.Tensor:
    return F.binary_cross_entropy(input, target.squeeze())

def accfn_binary_accuracy(input: t.Tensor, target: t.Tensor) -> t.Tensor:
    return binary_accuracy(input, target.squeeze())


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
        print(f"Figure of pianoroll is saved in '{save_path}'.")
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
        print(f"Figure of pianorolls is saved in '{save_path}'.")
    if is_show:
        plt.show()
    else:
        plt.close()

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
        print(f"Figure of loss and accuracy is saved in '{save_path}'.")
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
    print(f"Midi data is saved in '{save_path}{midext}'.")
    sf.write(f"{save_path}{wavext}", midi.synthesize(fs=sr), samplerate=sr)
    print(f"Rendered wave data is saved in '{save_path}{wavext}'.")

def plot_save_midi(dataloader: DataLoader, inference_fn: Callable[[t.Tensor], t.Tensor], logger: Logger,
                   hps: HyperParams, title: str, **plot_kwargs: Any) -> None:
    xs, _ = next(iter(dataloader))
    ys: PianoRollBatchTensor = inference_fn(xs)
    x: PianoRollTensor = xs[0, :, :-1]  # get first data, remove rest
    y: PianoRollTensor = ys[0, :, :-1]  # get first data, remove rest
    save_midi([x, y], logger=logger, title=title, note_offset=hps.data_note_low)
    plot_pianorolls(x, y, n_bars=hps.data_length_bars, note_low=hps.data_note_low, note_high=hps.data_note_high,
                    logger=logger, title=title, **plot_kwargs)