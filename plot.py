import pretty_midi as pm
import soundfile as sf
import matplotlib.pyplot as plt
from typedef import *
from hparam import HyperParams
from log import Logger


def plot_train_log(train_losses: TrainMetricLog, train_accs: TrainMetricLog,
                   test_losses: TrainMetricLog, test_accs: TrainMetricLog,
                   figsize: tuple[float, float] =(14, 5), is_show: bool =False,
                   is_save: bool =False, logger: Logger|None =None, title: str ="loss_acc") -> None:
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


def plot_pianoroll(pr: PianoRoll|PianoRollTensor, hps: HyperParams,
                   figsize: tuple[float, float] =(7, 5), is_show: bool =False,
                   is_save: bool =False, logger: Logger|None =None, title: str|None =None) -> None:
    if not is_save and not is_show:
        return

    _, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("bar")
    ax.set_ylabel("note")
    ax.imshow(pr.T, origin="lower", aspect="auto",
              extent=[0, hps.data_length_bars, hps.data_note_low, hps.data_note_high])  # type: ignore
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

def plot_pianorolls(pr0: PianoRoll|PianoRollTensor, pr1: PianoRoll|PianoRollTensor, hps: HyperParams,
                    figsize: tuple[float, float] =(14, 5), is_show: bool =False,
                    is_save: bool =False, logger: Logger|None =None, title: str|None =None) -> None:
    if not is_save and not is_show:
        return

    fig = plt.figure(figsize=figsize)
    axl = fig.add_subplot(1, 2, 1)
    axl.set_xlabel("bar")
    axl.set_ylabel("note")
    axl.imshow(pr0.T, origin="lower", aspect="auto",
              extent=[0, hps.data_length_bars, hps.data_note_low, hps.data_note_high])  # type: ignore

    axr = fig.add_subplot(1, 2, 2)
    axr.set_xlabel("bar")
    axr.set_ylabel("note")
    axr.imshow(pr1.T, origin="lower", aspect="auto",
              extent=[0, hps.data_length_bars, hps.data_note_low, hps.data_note_high])  # type: ignore
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


def save_midi(prl: list[PianoRoll|PianoRollTensor], logger: Logger,
              resolution: int =480, note_offset: int =0, sr: int =44100,
              title: str|None =None, midext: str =".mid", wavext: str =".wav") -> None:
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


def scatter(datas: list[LatentBatchTensor], labels: list[str], n_dim: int =2,
            figsize: tuple[float, float] =(7, 5), is_show: bool =False,
            is_save: bool =False, logger: Logger|None =None, title: str|None =None) -> None:
    assert len(datas) == len(labels)
    assert datas[0].shape[1] >= n_dim and n_dim in [2, 3]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=("rectilinear" if n_dim == 2 else "3d"))
    for i, (data, label) in enumerate(zip(datas, labels)):
        if n_dim == 2:
            ax.scatter(data[:, 0], data[:, 1], color=plt.get_cmap("tab10")(i), label=label)
        elif n_dim == 3:
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], color=plt.get_cmap("tab10")(i), label=label)
    plt.legend()
    plt.tight_layout()

    if is_save:
        assert logger is not None
        save_path: str = f"{logger.outdir}/{title if title is not None else f'scatter_{n_dim}d'}_{logger.time}.png"
        plt.savefig(save_path, dpi=320, bbox_inches="tight")
        logger(f"Figure of scatter {n_dim}D is saved in '{save_path}'.")
    if is_show:
        plt.show()
    else:
        plt.close()