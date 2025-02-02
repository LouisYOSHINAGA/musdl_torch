from typing import TypeAlias
import numpy as np
import torch as t

PianoRoll: TypeAlias = np.ndarray  # (time, note)
PianoRolls: TypeAlias = list[PianoRoll]  # [(time, note)]
PianoRollTensor: TypeAlias = t.Tensor  # (time, note)
PianoRollBatchTensor: TypeAlias = t.Tensor  # (batch, time, note)

TrainMetricLog: TypeAlias = list[float]

N_KEY_CLASS: int = 12