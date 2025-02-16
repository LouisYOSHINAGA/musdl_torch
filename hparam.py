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
    general_device="cpu",

    data_path="dataset",
    data_ext=".mid",
    data_is_sep_part=False,
    data_is_relative_pitch=False,
    data_resolution_nth_note=8,
    data_length_bars=8,
    data_note_low=36,
    data_note_high=84,
    data_extract_method="head",
    data_is_return_key_mode=False,
    data_batch_size=32,
    data_train_test_split=0.8,
    data_verbose=False,

    kdn_hidden_dims=[6, ],

    hrm_rnn_type="rnn",
    hrm_rnn_hidden_size=128,
    hrm_rnn_num_layers=1,
    hrm_rnn_bidirectional=False,
    hrm_rnn_dropout=0.0,

    train_lr=0.001,
    train_epochs=1000,
    train_save_period=1000,
    train_load_path=None,
    train_save_path=None,
    train_verbose=False,
)


def setup_hyperparams(**kwargs: Any) -> HyperParams:
    hps: HyperParams = default_hps
    for k, v in kwargs.items():
        assert k in hps, f"Hyper Parameter '{k}' does not exist."
        hps[k] = v
    return hps