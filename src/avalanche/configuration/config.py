import typing as t
from configparser import ConfigParser

from pydantic import BaseModel
from pydantic.validators import int_validator, str_validator


def empty_str_to_none(v: str) -> t.Optional[str]:
    if v == "":
        return None

    return str_validator(v)


def empty_int_to_none(v: str) -> t.Optional[str]:
    if v == "":
        return None
    return int_validator(v)


class EmptyStrToNone(str):
    @classmethod
    def __get_validators__(cls):
        yield empty_str_to_none


class EmptyIntToNone(str):
    @classmethod
    def __get_validators__(cls):
        yield empty_int_to_none


class BaseTrainConfig(BaseModel):
    # Training
    accelerator: str
    devices: t.Union[None, str, EmptyStrToNone]
    strategy: str = "auto"
    batch_size: int
    max_epochs: int
    min_epochs: int
    validate_every_n: int
    num_workers: int
    accumulate_grad_batches: t.Union[None, int, EmptyIntToNone]
    learning_rate: float
    dataset_path: str
    best_model_prefix: str

    # Logging
    evaluation_logger: str
    train_logger: str
    logging_path: str
    checkpoint_path: str = ""
    precision: str = "32-true"

    @staticmethod
    def construct_typed_config(ini_config: ConfigParser):
        """
        Creates typed version of ini configuration file

        :param ini_config: ConfigParser instance
        :return: Instance of TrainConfig
        """
        raise NotImplementedError
