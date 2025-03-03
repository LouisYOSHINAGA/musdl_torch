import os, logging
from datetime import datetime, timedelta, timezone
from hparam import HyperParams


class Logger:
    def __init__(self, hps: HyperParams) -> None:
        self.init_time()
        self.init_outdir(hps.general_output_path)
        self.init_logger(hps.general_log_path, hps.general_log_level)
        self.log_hps(hps)

    def init_time(self, fmt: str ="%Y%m%d_%H%M%S") -> None:
        self.time: str = datetime.now(timezone(timedelta(hours=9), "JST")).strftime(fmt)

    def init_outdir(self, outdir: str) -> None:
        assert os.path.isdir(outdir), f"Target directory '{outdir}' does not exist."
        self.outdir: str = f"{outdir}/out_{self.time}"
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
            print(f"Output directory '{self.outdir}' is newly made.")

    def init_logger(self, log_path: str|None, log_level: str) -> None:
        if log_path is None:
            log_path = f"{self.outdir}/log_{self.time}.log"
        log_level_dict: dict[str, int] = {
            'notset': logging.NOTSET,
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL,
        }

        self.logger = logging.getLogger("trainer")
        self.logger.setLevel(log_level_dict[log_level])
        self.logger.addHandler(logging.StreamHandler())
        self.logger.addHandler(logging.FileHandler(log_path))

        self.file_only_logger = logging.getLogger("trainer.file_only")
        self.file_only_logger.setLevel(log_level_dict[log_level])
        self.file_only_logger.propagate = False
        self.file_only_logger.addHandler(logging.FileHandler(log_path))

    def log_hps(self, hps: HyperParams) -> None:
        self.logger.info(f"Hyper Parameters:")
        for k, v in hps.items():
            self.logger.info(f"    {k}: {v}")
        self.logger.info("")

    def __call__(self, msg: str, is_file_only: bool =False) -> None:
        self.info(msg, is_file_only)

    def debug(self, msg: str, is_file_only: bool =False) -> None:
        if is_file_only:
            self.file_only_logger.debug(msg)
        else:
            self.logger.debug(msg)

    def info(self, msg: str, is_file_only: bool =False) -> None:
        if is_file_only:
            self.file_only_logger.info(msg)
        else:
            self.logger.info(msg)

    def warning(self, msg: str, is_file_only: bool =False) -> None:
        if is_file_only:
            self.file_only_logger.warning(msg)
        else:
            self.logger.warning(msg)

    def error(self, msg: str, is_file_only: bool =False) -> None:
        if is_file_only:
            self.file_only_logger.error(msg)
        else:
            self.logger.error(msg)

    def critical(self, msg: str, is_file_only: bool =False) -> None:
        if is_file_only:
            self.file_only_logger.critical(msg)
        else:
            self.logger.critical(msg)


def setup_logger(hps: HyperParams) -> Logger:
    return Logger(hps)