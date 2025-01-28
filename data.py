import os, glob
import pretty_midi as pm
import numpy as np
import torch as t
from torch.utils.data import Dataset
from hparam import HyperParams
from typing import TypeAlias

PianoRoll: TypeAlias = np.ndarray  # (time, note)
PianoRolls: TypeAlias = list[PianoRoll]


class MIDIChoraleDatset(Dataset):
    def __init__(self, hps: HyperParams) -> None:
        self.is_sep_part: bool = hps.data_is_sep_part
        self.sequece_length: int = hps.data_resolution_nth_note * hps.data_length_bars
        self.verbose: bool = hps.data_verbose
        self.load(hps)

    def load(self, hps: HyperParams) -> None:
        assert os.path.isdir(hps.data_path), f"Target directory '{hps.data_path}' does not exist."

        self.prs: PianoRolls = []
        self.prs_sop: PianoRolls = []
        self.prs_alt: PianoRolls = []
        key_modes: list[int] = []

        for f in glob.glob(f"{hps.data_path}/*{hps.data_ext}"):
            prs_dct, key_mode = self.load_midi_file(
                f, is_relative_pitch=hps.data_is_relative_pitch, resolution=hps.data_resolution_nth_note
            )
            if prs_dct is None or key_mode is None:
                continue
            if self.is_sep_part:
                self.prs_sop.append(prs_dct["sop"])
                self.prs_alt.append(prs_dct["alt"])
            else:
                self.prs.append(prs_dct["whole"])
            key_modes.append(key_mode)
        self.key_modes: np.ndarray = np.array(key_modes)

    def load_midi_file(self, filename: str, is_relative_pitch: bool, resolution: int) \
        -> tuple[dict[str, PianoRoll], int] | tuple[None, None]:

        if not os.path.isfile(filename):
            return None, None
        midi: pm.PrettyMIDI = pm.PrettyMIDI(filename)
        self.vprint(f"Loading MIDI file '{filename}'.")

        if self.is_sep_part and len(midi.instruments) < 2:  # use songs with 2 or more parts
            self.vprint(f"Skip loading: lack of parts ({len(midi.instruments)} parts); '{filename}'")
            return None, None
        if len(midi.key_signature_changes) != 1:  # use songs without modulation
            self.vprint(f"Skip loading: modulation included ({len(midi.instruments)} times); '{filename}'")
            return None, None
        if len(midi.get_tempo_changes()[1]) != 1:  # user songs without tempo change
            self.vprint(f"Skip loading: tempo change included ({len(midi.get_tempo_changes()[1])} times); '{filename}'")
            return None, None

        key_number: int = midi.key_signature_changes[0].key_number  # key number \in {major: {0..11}, minor: {12..23}}
        key_mode: int = key_number // 12  # major => 0, minor => 1
        tempo: int = midi.get_tempo_changes()[1][0]

        if is_relative_pitch:
            self.vprint(f"Songs are transposed to C major/minor.")
            midi = self.transpose(midi, key_number)

        prs_dct: dict[str, PianoRoll] = {
            'sop': self.midi_to_pianoroll(midi.instruments[0], tempo=tempo, resolution=resolution),
            'alt': self.midi_to_pianoroll(midi.instruments[1], tempo=tempo, resolution=resolution)
        } if self.is_sep_part else {
            'whole': self.midi_to_pianoroll(midi, tempo=tempo, resolution=resolution)
        }
        return prs_dct, key_mode

    def vprint(self, msg: str) -> None:
        if self.verbose:
            print(f"{msg}")

    def transpose(self, midi: pm.PrettyMIDI, key_number: int) -> pm.PrettyMIDI:
        for inst in midi.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    note.pitch -= key_number % 12
        return midi

    def midi_to_pianoroll(self, midi: pm.PrettyMIDI|pm.Instrument, tempo: int, resolution: int,
                          note_low: int =36, note_high: int=84) -> PianoRoll:
        # tempo: beats per minute = 4th-notes per 60 seconds
        # -> tempo/60: sample same number of times as 4th-notes per second
        # -> (N/4)*tempo/60: sample same number of times as Nth-notes per second
        pr: PianoRoll = midi.get_piano_roll(fs=int((resolution/4)*tempo/60))  # (pitch, time)
        if pr.shape[1] < self.sequece_length:
            pr = np.concatenate([np.zeros((pr.shape[0], self.sequece_length-pr.shape[1])), pr], axis=1)
        return np.where(pr[note_low:note_high] <= 0, 0, 1).T

    def __getitem__(self, index: int) -> t.Tensor:
        ...

    def __len__(self) -> int:
        return len(self.key_modes)


if __name__ == "__main__":
    from hparam import setup_hyperparams
    dataset: Dataset = MIDIChoraleDatset(setup_hyperparams(data_verbose=True))
    print(len(dataset))