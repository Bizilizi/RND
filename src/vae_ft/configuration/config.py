from configparser import ConfigParser

from src.avalanche.configuration.config import BaseTrainConfig


class TrainConfig(BaseTrainConfig):
    # Model
    input_dim: int
    z_dim: int
    model_backbone: str
    num_random_images: int
    num_random_noise: int

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
            **ini_config["model"],
        )

        return config
