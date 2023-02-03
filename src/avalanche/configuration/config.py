from configparser import ConfigParser

from pydantic import BaseModel
import typing as t
from pydantic.validators import str_validator


def empty_to_none(v: str) -> t.Optional[str]:
    if v == "":
        return None
    return v


class EmptyStrToNone(str):
    @classmethod
    def __get_validators__(cls):
        yield str_validator
        yield empty_to_none


class BaseTrainConfig(BaseModel):
    # Training
    accelerator: str
    devices: str
    batch_size: int
    max_epochs: int
    min_epochs: int
    validate_every_n: int
    num_workers: int
    accumulate_grad_batches: t.Union[None, int, EmptyStrToNone]
    learning_rate: float

    # Logging
    evaluation_logger: str
    train_logger: str
    logging_path: str

    @staticmethod
    def construct_typed_config(ini_config: ConfigParser):
        """
        Creates typed version of ini configuration file

        :param ini_config: ConfigParser instance
        :return: Instance of TrainConfig
        """
        raise NotImplementedError
