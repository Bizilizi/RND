from configparser import ConfigParser

from src.avalanche.configuration.config import BaseTrainConfig


class TrainConfig(BaseTrainConfig):
    # Model
    num_embeddings: int
    add_embeddings_per_step: int
    enc_embedding_dim: int
    img_embedding_dim: int
    commitment_cost: float
    decay: float
    num_tasks: int
    num_epochs_schedule: str
    bootstrapped_dataset_path: str
    dataset: str
    dataset_variance: float

    mask_ratio: float = 0.75
    weight_decay: float
    latent_consistency_sigma: float

    # weight
    l1_loss_weight: float
    lpip_loss_weight: float
    vq_loss_weight: float
    latent_consistency_loss_weight: float
    discriminator_weight: float

    # sampling
    num_random_past_samples: int
    num_random_past_samples_schedule: str
    temperature: float

    # gpt
    gpt_num_layers: int
    gpt_num_epochs_max: int
    gpt_num_epochs_min: int
    gpt_batch_size: int
    gpt_accumulate_grad_batches: int
    gpt_learning_rate: float
    gpt_mask_ratio: float
    gpt_mask_token_weight: float

    # Classifier
    classifier_max_epochs: int
    classifier_min_epochs: int
    classifier_batch_size: int

    @staticmethod
    def construct_typed_config(ini_config: ConfigParser) -> "TrainConfig":
        """
        Creates typed version of ini configuration file

        :param ini_config: ConfigParser instance
        :return: Instance of TrainConfig
        """

        config = TrainConfig(
            **ini_config["qmae"],
            **ini_config["classifier"],
            **ini_config["sampling"],
            **ini_config["gpt"],
            **ini_config["training"],
            **ini_config["logging"],
        )

        return config
