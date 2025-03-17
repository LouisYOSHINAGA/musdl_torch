import os, glob, math, re
import pretty_midi as pm
import numpy as np
import torch as t
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
from typing import Any
from typedef import *
from hparam import HyperParams
from log import Logger

Data: TypeAlias = tuple[PianoRollTensor, NoteSequenceTensor, KeyModeTensor] \
                | tuple[PianoRollTensor, KeyModeTensor]


class MIDIChoraleDatset(Dataset):
    def __init__(self, hps: HyperParams, logger: Logger) -> None:
        self.is_sep_part: bool = hps.data_is_sep_part
        self.sequence_length: int = hps.data_resolution_nth_note * hps.data_length_bars
        self.extract_method: str = hps.data_extract_method
        self.rng: np.random.Generator = np.random.default_rng()
        self.is_recons: bool = hps.data_is_recons
        self.batch_size: int = hps.data_batch_size
        self.logger: Logger = logger
        self.load(hps)

    def load(self, hps: HyperParams) -> None:
        assert os.path.isdir(hps.data_path), f"Target directory '{hps.data_path}' does not exist."

        self.prs_sop: PianoRolls = []
        self.prs_alt: PianoRolls = []
        self.prs: PianoRolls = []
        self.key_modes: list[int] = []
        self.filenames: list[str] = []

        for f in glob.glob(f"{hps.data_path}/*{hps.data_ext}"):
            prs_dct, key_mode = self.load_midi_file(
                f, is_relative_pitch=hps.data_is_relative_pitch, resolution=hps.data_resolution_nth_note,
                note_low=hps.data_note_low, note_high=hps.data_note_high
            )
            if prs_dct is None or key_mode is None:  # invalid midi file
                continue

            if self.is_sep_part:
                self.prs_sop.append(prs_dct["sop"])
                self.prs_alt.append(prs_dct["alt"])
            else:
                self.prs.append(prs_dct["all"])
            self.key_modes.append(key_mode)
            self.filenames.append(os.path.basename(f))

        self.logger(f"Load dataset with {len(self.key_modes)} samples from '{hps.data_path}'.")
        if hps.data_is_relative_pitch:
            self.logger(f"Songs are transposed to C major/minor.")

    def load_midi_file(self, filename: str, is_relative_pitch: bool, resolution: int, \
                       note_low: int, note_high: int) -> tuple[dict[str, PianoRoll], int] | tuple[None, None]:
        if not os.path.isfile(filename):
            return None, None
        midi: pm.PrettyMIDI = pm.PrettyMIDI(filename)
        self.logger.debug(f"Loading MIDI file '{filename}'.")

        if self.is_sep_part and len(midi.instruments) < 2:  # use songs with 2 or more parts
            self.logger.debug(f"Skip loading '{filename}': lack of parts ({len(midi.instruments)} parts).")
            return None, None

        if len(midi.key_signature_changes) != 1:  # use songs without modulation
            self.logger.debug(f"Skip loading '{filename}': modulation included ({len(midi.instruments)} times).")
            return None, None
        key_number: int = midi.key_signature_changes[0].key_number  # key number \in {major: {0..11}, minor: {12..23}}
        key_mode: int = key_number // N_KEY_CLASS  # major => 0, minor => 1

        if len(midi.get_tempo_changes()[1]) != 1:  # user songs without tempo change
            self.logger.debug(f"Skip loading '{filename}: tempo change included ({len(midi.get_tempo_changes()[1])} times).")
            return None, None
        tempo: int = midi.get_tempo_changes()[1][0]

        if is_relative_pitch:
            midi = self.transpose(midi, key_number)

        prs_dct: dict[str, PianoRoll] = {
            'sop': self.midi_to_pianoroll(
                midi.instruments[0], tempo=tempo, resolution=resolution, note_low=note_low, note_high=note_high
            ),
            'alt': self.midi_to_pianoroll(
                midi.instruments[1], tempo=tempo, resolution=resolution, note_low=note_low, note_high=note_high
            ),
        } if self.is_sep_part else {
            'all': self.midi_to_pianoroll(
                midi, tempo=tempo, resolution=resolution, note_low=note_low, note_high=note_high
            ),
        }
        return prs_dct, key_mode

    def transpose(self, midi: pm.PrettyMIDI, key_number: int) -> pm.PrettyMIDI:
        for inst in midi.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    note.pitch -= key_number % N_KEY_CLASS
        return midi

    def midi_to_pianoroll(self, midi: pm.PrettyMIDI|pm.Instrument,
                          tempo: int, resolution: int, note_low: int, note_high: int) -> PianoRoll:
        # tempo: beats per minute = 4th-notes per 60 seconds
        # -> tempo/60: sample same number of times as 4th-notes per second
        # -> (N/4)*tempo/60: sample same number of times as Nth-notes per second
        pr: PianoRoll = midi.get_piano_roll(fs=int((resolution/4)*tempo/60))  # (pitch, time)
        if pr.shape[1] < self.sequence_length:
            pr = np.concatenate([pr, np.zeros((pr.shape[0], self.sequence_length-pr.shape[1]))], axis=1)
        return np.where(pr[note_low:note_high] <= 0, 0, 1).T

    def __getitem__(self, index: int) -> tuple[str, Data]:
        kmt: KeyModeTensor = t.Tensor([self.key_modes[index]])
        if self.is_sep_part:
            pr_sop: PianoRoll = self.prs_sop[index]
            pr_recons: PianoRoll = pr_sop if self.is_recons else self.prs_alt[index]
            start, end = self.get_sequence_range(full_length=pr_sop.shape[0])
            data: Data = self.onehot(pr_sop[start:end]), self.numerical(pr_recons[start:end]), kmt
        else:
            pr: PianoRoll = self.prs[index]
            start, end = self.get_sequence_range(full_length=pr.shape[0])
            data = t.Tensor(pr[start:end]), kmt
        return self.filenames[index], data

    def get_sequence_range(self, full_length: int) -> tuple[int, int]:
        first_half_length: int = math.floor(self.sequence_length/2)
        second_half_length: int = math.ceil(self.sequence_length/2)
        if self.extract_method == "head":
            start: int = 0
            end: int = start + self.sequence_length
        elif self.extract_method == "center":
            center: int = math.floor(full_length/2)
            start = center - first_half_length
            end = center + second_half_length
        elif self.extract_method == "tail":
            end  = full_length
            start = end - self.sequence_length
        elif self.extract_method == "random":
            center = first_half_length if self.sequence_length == full_length \
                     else self.rng.integers(low=first_half_length, high=full_length-second_half_length)
            start = center - first_half_length
            end = center + second_half_length
        else:
            assert False, f"Unexpected method of data sequence extraction."
        return start, end

    def onehot(self, pr: PianoRoll) -> PianoRollTensor:
        is_rest: np.ndarray = np.expand_dims(1 - np.sum(pr, axis=1), axis=1)
        return t.Tensor(np.concatenate([pr, is_rest], axis=1))

    def numerical(self, pr: PianoRoll) -> NoteSequenceTensor:
        return t.argmax(self.onehot(pr), dim=1)

    def __len__(self) -> int:
        return len(self.key_modes)


class MIDIChoraleCollator:
    def __init__(self, conf: str) -> None:
        self.is_filename: bool = False
        self.is_keymode: bool = False
        self.set_modes(conf)

    def __call__(self, batch: list[tuple[str, Data]]) -> tuple[list[str], t.Tensor]|t.Tensor:
        return ([b[0] for b in batch], self.collate(batch)) if self.is_filename else self.collate(batch)

    def collate(self, batch: list[tuple[str, Data]]) -> t.Tensor:
        if self.is_keymode:  # (..., km)
            return default_collate([b[1] for b in batch])
        else:
            if len(batch[0][1]) == 2:  # (pr, km) -> pr
                return default_collate([b[1][0] for b in batch])
            else:  # (pr_sop, pr_recons, km) -> (pr_sop, pr_recons)
                return default_collate([b[1][:-1] for b in batch])

    def set_mode(self, key: str, value: bool) -> None:
        if key == 'f' or key == "filename":
            self.is_filename = value
        elif key == 'k' or key == "keymode":
            self.is_keymode = value

    def set_modes(self, conf: str) -> None:
        assert re.fullmatch(r"(!?(f|k))*", conf), f"Invalid syntax."
        value: bool = True
        for c in conf:
            if c == '!':
                value = False
            else:
                self.set_mode(c, value)
                value = True


class MIDIChoraleDataLoader(DataLoader):
    def __init__(self, conf: str, **kwargs: Any) -> None:
        self.dynamic_collator: MIDIChoraleCollator = MIDIChoraleCollator(conf)
        super().__init__(**kwargs, collate_fn=self.dynamic_collator)

    def set_mode(self, key: str, value: bool) -> None:
        self.dynamic_collator.set_mode(key, value)

    def set_modes(self, conf: str) -> None:
        self.dynamic_collator.set_modes(conf)


def setup_dataloaders(hps: HyperParams, logger: Logger, conf: str) -> tuple[MIDIChoraleDataLoader, MIDIChoraleDataLoader]:
    dataset: Dataset = MIDIChoraleDatset(hps, logger)

    assert 0 <= hps.data_train_test_split < 1, \
        f"Invalid train:test split rate; '{hps.data_train_test_split}' must be in [0, 1)."
    n_data: int = len(dataset)
    n_train_data: int = int(hps.data_train_test_split * n_data)
    n_test_data: int = n_data - n_train_data
    logger(f"Split the dataset into training data ({n_train_data} samples) and test data ({n_test_data} samples).\n")

    train_dataset, test_dataset = random_split(dataset, [n_train_data, n_test_data])
    train_dataloader = MIDIChoraleDataLoader(dataset=train_dataset, batch_size=hps.data_batch_size, shuffle=True, conf=conf)
    test_dataloader = MIDIChoraleDataLoader(dataset=test_dataset, batch_size=hps.data_batch_size, shuffle=False, conf=conf)
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    from hparam import setup_hyperparams
    from log import setup_logger
    from plot import plot_pianoroll

    hps: HyperParams = setup_hyperparams(data_is_sep_part=False, general_log_level="debug")
    logger: Logger = setup_logger(hps)
    _, test_dataloader = setup_dataloaders(hps, logger, conf="f!k")

    fns, prbt = next(iter(test_dataloader))
    print(f"Batch size = {prbt.shape}")

    print(f"Plot pianoroll: {fns[0]}")
    plot_pianoroll(prbt[0], hps=hps, is_show=True)