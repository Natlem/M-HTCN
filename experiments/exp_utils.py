from dotenv import load_dotenv
import os
from typing import Any
from dataclasses import dataclass

@dataclass
class Args:
    dataset: str = ''
    dataset_t: Any = None
    set_cfgs: list = None
    imdb_name_target: Any = None
    imdb_name: str = ''
    large_scale: bool = False
    cfg_file: str = None
    net: str = ''
    disp_interval: int = 100
    eta: int = 0.1

class LoggerForSacred():
    def __init__(self, visdom_logger, ex_logger=None, always_print=True):
        self.visdom_logger = visdom_logger
        self.ex_logger = ex_logger
        self.always_print = always_print


    def log_scalar(self, metrics_name, value, step):
        if self.visdom_logger is not None:
            self.visdom_logger.scalar(metrics_name, step, [value])
        if self.ex_logger is not None:
            self.ex_logger.log_scalar(metrics_name, value, step)
        if self.always_print:
            print("{}:{}/{}".format(metrics_name, value, step))


def get_config_var(is_eval=False):
    if not os.path.exists('.env'):
        raise FileNotFoundError('.env not found')

    load_dotenv()

    vars = {}
    vars["SACRED_URL"] = os.getenv("SACRED_URL")
    vars["SACRED_DB"] = os.getenv("SACRED_DB") if not is_eval else os.getenv("SACRED_DB")  + '_eval'
    vars["VISDOM_PORT"] = os.getenv("VISDOM_PORT")
    vars["SAVE_DIR"] = os.getenv("SAVE_DIR")
    vars["GMAIL_USER"] = os.getenv("GMAIL_USER")
    vars["GMAIL_PASSWORD"] = os.getenv("GMAIL_PASSWORD")
    vars["TO_EMAIL"] = os.getenv("TO_EMAIL")
    vars["SACRED_USER"] = os.getenv("SACRED_USER")
    vars["SACRED_PWD"] = os.getenv("SACRED_PWD")

    if not os.path.exists(vars["SAVE_DIR"]):
        os.makedirs(vars["SAVE_DIR"])

    return vars

all_envs =  get_config_var()
all_save_dir = all_envs["SAVE_DIR"]