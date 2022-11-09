from configparser import ConfigParser

from pydantic import BaseModel


class TrainConfig(BaseModel):
    # Training
    gpus: str
    batch_size: int
    max_epochs: int
    validate_every_n: int
    num_workers: int
    accumulate_grad_batches: int
    learning_rate: float

    # Logging
    logger_type: str
    logging_path: str

    @staticmethod
    def construct_typed_config(ini_config: ConfigParser) -> "TrainConfig":
        """
        Creates typed version of ini configuration file

        :param ini_config: ConfigParser instance
        :return: Instance of TrainConfig
        """

        config = TrainConfig(
            **ini_config["training"],
            **ini_config["logging"],
        )

        return config
