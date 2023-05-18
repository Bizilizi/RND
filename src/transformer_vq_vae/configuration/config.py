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

    regularization_lambda: float
    regularization_dropout: float

    bootstrapped_dataset_path: str

    use_lpips: bool
    mask_ratio: float
    weight_decay: float
    embeddings_distance: str

    # sampling
    num_random_future_samples: int
    num_random_past_samples: int
    future_samples_mode: str
    sampling_temperature: float

    # igpt
    num_gpt_layers: int
    igpt_batch_size: int
    igpt_mask_ratio: float
    igpt_accumulate_grad_batches: int
    igpt_mask_token_weight: float

    # loss weights
    vq_loss_weight = 1
    reconstruction_loss_weight = 1
    contrastive_loss_loss_weight = 1
    encoder_mlm_loss_loss_weight = 1
    decoder_regression_loss_loss_weight = 1

    # training
    max_epochs_lin_eval: int
    min_epochs_lin_eval: int

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
            **ini_config["sampling"],
            **ini_config["igpt"]
        )

        return config
