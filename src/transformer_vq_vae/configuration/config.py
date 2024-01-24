from configparser import ConfigParser

from src.avalanche.configuration.config import BaseTrainConfig


class TrainConfig(BaseTrainConfig):
    # Model
    num_embeddings: int
    num_embeddings_per_step: int
    num_class_embeddings: int
    embedding_dim: int
    commitment_cost: float
    decay: float
    bootstrapped_dataset_path: str

    encoder_layer: int
    encoder_head: int
    decoder_layer: int
    decoder_head: int

    patch_size: int = 4
    supervised: bool = False

    use_lpips: bool
    mask_ratio: float
    weight_decay: float

    # loss coefficients
    cycle_consistency_loss_weight_for_past: float
    cycle_consistency_loss_weight_for_current: float
    cycle_consistency_sigma: float

    past_samples_loss_weight: float
    current_samples_loss_weight: float
    future_samples_loss_weight: float
    data_variance: float

    # quantization params
    quantize_features: bool
    quantize_top_k: int
    perplexity_threshold: float

    # sampling
    num_random_future_samples: int
    num_random_past_samples: int
    num_random_past_samples_schedule: str
    future_samples_mode: str
    temperature: float
    reuse_igpt: bool

    # igpt
    num_gpt_layers: int
    igpt_batch_size: int
    igpt_mask_ratio: float
    igpt_accumulate_grad_batches: int
    igpt_mask_token_weight: float

    # training (the rest in the BaseTrainConfig class)
    dataset: str = "cifar10"
    num_tasks: int
    image_size: int = 32

    max_epochs_lin_eval: int
    min_epochs_lin_eval: int

    warmup: float

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
