from configparser import ConfigParser

from pydantic import BaseModel


class BaseTrainConfig(BaseModel):
    # Training
    accelerator: str
    devices: str
    batch_size: int
    max_epochs: int
    min_epochs: int
    validate_every_n: int
    num_workers: int
    accumulate_grad_batches: int
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
