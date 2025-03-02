import os, logging
from datetime import datetime, timedelta, timezone
from hparam import HyperParams


class Logger:
    def __init__(self, hps: HyperParams) -> None:
        self.init_time()
        self.init_outdir(hps.general_output_path)
        self.init_logger(hps.general_log_path)
        self.log_hps(hps)

    def init_time(self, fmt: str ="%Y%m%d_%H%M%S") -> None:
        self.time: str = datetime.now(timezone(timedelta(hours=9), "JST")).strftime(fmt)

    def init_outdir(self, outdir: str) -> None:
        assert os.path.isdir(outdir), f"Target directory '{outdir}' does not exist."
        self.outdir: str = f"{outdir}/out_{self.time}"
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
            print(f"Output directory '{self.outdir}' is newly made.")

    def init_logger(self, log_path: str|None) -> None:
        self.logger = logging.getLogger("trainer")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.addHandler(logging.FileHandler(
            log_path if log_path is not None else f"{self.outdir}/log_{self.time}.log"
        ))

    def log_hps(self, hps: HyperParams) -> None:
        self.logger.info(f"Hyper Parameters:")
        for k, v in hps.items():
            self.logger.info(f"    {k}: {v}")
        self.logger.info("")

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def __call__(self, msg: str) -> None:
        self.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        self.logger.critical(msg)


def setup_logger(hps: HyperParams) -> Logger:
    return Logger(hps)