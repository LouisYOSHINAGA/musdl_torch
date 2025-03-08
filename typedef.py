from typing import TypeAlias, Callable
import numpy as np
import torch as t

PianoRoll: TypeAlias = np.ndarray  # (time, note)
PianoRolls: TypeAlias = list[PianoRoll]  # [(time, note)]
PianoRollTensor: TypeAlias = t.Tensor  # (time, note)
PianoRollBatchTensor: TypeAlias = t.Tensor  # (batch, time, note)
NoteSequenceTensor: TypeAlias = t.Tensor  # (time, )
NoteSequenceBatchTensor: TypeAlias = t.Tensor  # (batch, time)

Optimizer: TypeAlias = t.optim.SGD | t.optim.Adagrad | t.optim.RMSprop | t.optim.Adam
CriterionFn: TypeAlias = Callable[[t.Tensor, t.Tensor], t.Tensor] \
                       | Callable[[tuple[t.Tensor, t.Tensor], t.Tensor], t.Tensor]
TrainMetricLog: TypeAlias = list[float]

N_KEY_CLASS: int = 12
KEY_MAJOR: int = 0
KEY_MINOR: int = 1