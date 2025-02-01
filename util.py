import matplotlib.pyplot as plt
from typedef import PianoRoll, PianoRollTensor


def plot_pianoroll(pr: PianoRoll|PianoRollTensor, n_bars: int,
                   figsize: tuple[float, float] =(8, 6), note_low: int =36, note_high: int=84) -> None:
    _, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("bar")
    ax.set_ylabel("note")
    ax.imshow(pr.T, aspect="auto", extent=[0, n_bars, note_low, note_high])  # type: ignore
    plt.show()