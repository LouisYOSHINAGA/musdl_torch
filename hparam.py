import numpy as np
from typing import Any


class HyperParams(dict):
    def __getattr__(self, key: str) -> Any:
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state


default_hps = HyperParams(
    data_path="dataset",
    data_ext=".mid",
    data_is_relative_pitch=False,
    data_resolution_nth_note=8,
    data_length_bars=8,
    data_is_sep_part=False,
    data_verbose=False,
)


def setup_hyperparams(**kwargs: Any) -> HyperParams:
    hps: HyperParams = default_hps
    for k, v in kwargs.items():
        assert k in hps, f"Hyper Parameter '{k}' does not exist."
        hps[k] = v
    return hps