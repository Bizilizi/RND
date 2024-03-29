from configparser import ConfigParser

from pydantic import BaseModel

from src.avalanche.configuration.config import BaseTrainConfig


class TrainConfig(BaseTrainConfig):
    # Model
    image_generation_batch_size: int
    input_dim: int
    num_random_images: int
    num_generation_attempts: int = 20
    l2_threshold: float
    rnd_latent_dim: int
    generator_type: str
    generator_checkpoint: str

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
