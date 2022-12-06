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
    evaluation_logger: str
    train_logger: str
    logging_path: str

    # Model
    image_generation_batch_size: int
    input_dim: int
    num_random_images: int
    l2_threshold: float
    rnd_latent_dim: int
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
