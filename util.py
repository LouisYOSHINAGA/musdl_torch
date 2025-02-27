import os
from datetime import datetime, timedelta, timezone
import pretty_midi as pm
import soundfile as sf
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Callable
from typedef import *
from hparam import HyperParams


def rnn_general(rnn_type: str, **rnn_kwargs: Any) -> nn.RNNBase:
    rnns: dict[str, type[nn.RNNBase]] = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
    assert rnn_type in rnns.keys(), f"Unexpected RNN type '{rnn_type}'. Available RNN type: {list(rnns.keys())}."
    return rnns[rnn_type](**rnn_kwargs)


class Logger:
    def __init__(self, hps: HyperParams) -> None:
        self.init_time()
        self.init_outdir(hps.general_output_path)

    def init_time(self, fmt: str ="%Y%m%d_%H%M%S") -> None:
        self.time: str = datetime.now(timezone(timedelta(hours=9), "JST")).strftime(fmt)

    def init_outdir(self, outdir: str) -> None:
        assert os.path.isdir(outdir), f"Target directory '{outdir}' does not exist."
        self.outdir: str = f"{outdir}/out_{self.time}"
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
            print(f"Output directory '{self.outdir}' is newly made.")


def plot_pianoroll(pr: PianoRoll|PianoRollTensor, n_bars: int,
                   note_low: int, note_high: int, figsize: tuple[float, float] =(7, 5)) -> None:
    _, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("bar")
    ax.set_ylabel("note")
    ax.imshow(pr.T, origin="lower", aspect="auto", extent=[0, n_bars, note_low, note_high])  # type: ignore
    plt.tight_layout()
    plt.show()

def plot_pianorolls(pr0: PianoRoll|PianoRollTensor, pr1: PianoRoll|PianoRollTensor, n_bars: int,
                    note_low: int, note_high: int, figsize: tuple[float, float] =(14, 5)) -> None:
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
    plt.show()

def plot_train_log(train_losses: TrainMetricLog, train_accs: TrainMetricLog,
                   test_losses: TrainMetricLog, test_accs: TrainMetricLog,
                   figsize: tuple[float, float] =(14, 5)) -> None:
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
    plt.show()

def save_midi(prl: list[PianoRoll|PianoRollTensor], logger: Logger, filename: str|None =None,
              midext: str =".mid", wavext: str =".wav", note_offset: int =0, resolution: int =480, sr: int =44100) -> None:
    save_path: str = f"{logger.outdir}/midi_{logger.time}" if filename is None else f"{logger.outdir}/{filename}"

    midi: pm.PrettyMIDI = pm.PrettyMIDI(resolution=resolution)
    for pr in prl:
        assert len(pr.shape) == 2  # (time, note)
        inst: pm.Instrument = pm.Instrument(program=1)
        for i in range(pr.shape[0]):
            for j in range(pr.shape[1]):
                if pr[i, j] > 0.5:
                    inst.notes.append(pm.Note(start=i/2, end=(i+1)/2, pitch=j+note_offset, velocity=100))
        midi.instruments.append(inst)
    midi.write(f"{save_path}{midext}")
    sf.write(f"{save_path}{wavext}", midi.synthesize(fs=sr), samplerate=sr)

def plot_save_midi(dataloader: DataLoader, inference_fn: Callable[[t.Tensor], t.Tensor], logger: Logger,
                   hps: HyperParams, filename: str|None =None) -> None:
    xs, _ = next(iter(dataloader))
    ys: PianoRollBatchTensor = inference_fn(xs)
    x: PianoRollTensor = xs[0, :, :-1]  # get first data, remove rest
    y: PianoRollTensor = ys[0, :, :-1]  # get first data, remove rest
    save_midi([x, y], logger=logger, filename=filename, note_offset=hps.data_note_low)
    plot_pianorolls(x, y, n_bars=hps.data_length_bars,
                    note_low=hps.data_note_low, note_high=hps.data_note_high)