import os
import os.path as osp
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime


class TruncatingFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        full = super().formatTime(record, datefmt)
        return full[11:19]

    def format(self, record):
        record.levelname = record.levelname[:4]
        return super().format(record)


def log_file(log_dir: str = "./work_dir/logs", file_name: str = None, level: int = logging.INFO):

    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    root_logger.setLevel(level)

    os.makedirs(log_dir, exist_ok=True)

    if file_name is None:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{current_datetime}.log"

    log_file_path = osp.join(log_dir, file_name)

    formatter = logging.Formatter(
        "[%(asctime)-19s]\t[%(levelname)-8s]\t[%(filename)-20s]"
        + "(%(funcName)-20s %(lineno)4d): %(message)s"
    )
    file_log_handler = RotatingFileHandler(
        log_file_path, maxBytes=10 * 1024 * 1024, backupCount=10
    )
    file_log_handler.setFormatter(formatter)
    root_logger.addHandler(file_log_handler)

    console_formatter = TruncatingFormatter(
        "[%(asctime)s %(levelname)s] %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging Handler Set Successful to: {log_file_path} & level={level}")
