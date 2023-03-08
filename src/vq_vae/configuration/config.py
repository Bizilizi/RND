from configparser import ConfigParser

from src.avalanche.configuration.config import BaseTrainConfig


class TrainConfig(BaseTrainConfig):
    # Model
    num_hiddens: int
    num_residual_layers: int
    num_residual_hiddens: int
    num_embeddings: int
    embedding_dim: int
    commitment_cost: float
    decay: float

    num_random_noise: int
    regularization_lambda: float
    regularization_dropout: float

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
