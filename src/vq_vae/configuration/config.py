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

    num_random_past_samples: int

    num_random_future_samples: int
    future_samples_mode: str
    sampling_temperature: float
    num_gpt_layers: int

    regularization_lambda: float
    regularization_dropout: float

    use_lpips: bool
    corruption_rate = 0.2

    embeddings_distance: str

    # loss weights
    vq_loss_weight = 1
    reconstruction_loss_weight = 1
    downstream_loss_weight = 1

    # training
    max_epochs_lin_eval: int
    min_epochs_lin_eval: int

    max_epochs_igpt: int
    min_epochs_igpt: int

    # training
    bootstrapped_dataset_path: str

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
