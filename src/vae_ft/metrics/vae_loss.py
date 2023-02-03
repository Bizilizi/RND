import typing as t

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics import ExperienceLoss, StreamLoss


class VAEExperienceLoss(ExperienceLoss):
    def __init__(
        self, with_kl_loss: bool = True, with_reconstruction_loss: bool = True
    ):
        super().__init__()

        self.with_kl_loss = with_kl_loss
        self.with_reconstruction_loss = with_reconstruction_loss

    def update(self, strategy):
        # task labels defined for each experience
        if hasattr(strategy.experience, "task_labels"):
            task_labels = strategy.experience.task_labels
        else:
            task_labels = [0]  # add fixed task label if not available.

        if len(task_labels) > 1:
            # task labels defined for each pattern
            # fall back to single task case
            task_label = 0
        else:
            task_label = task_labels[0]

        kl_div, reconstruction_loss = strategy.loss
        loss = 0

        if self.with_kl_loss:
            loss -= kl_div

        if self.with_reconstruction_loss:
            loss -= reconstruction_loss

        self._loss.update(loss, patterns=len(strategy.mb_y), task_label=task_label)

    def __str__(self):
        if self.with_kl_loss and self.with_reconstruction_loss:
            return "test/loss_exp"

        elif self.with_reconstruction_loss:
            return "test/reconstruction_loss_exp"
        else:
            return "test/kl_loss_exp"


class VAEStreamLoss(StreamLoss):
    def __init__(
        self, with_kl_loss: bool = True, with_reconstruction_loss: bool = True
    ):
        super().__init__()

        self.with_kl_loss = with_kl_loss
        self.with_reconstruction_loss = with_reconstruction_loss

    def update(self, strategy):
        # task labels defined for each experience
        if hasattr(strategy.experience, "task_labels"):
            task_labels = strategy.experience.task_labels
        else:
            task_labels = [0]  # add fixed task label if not available.

        if len(task_labels) > 1:
            # task labels defined for each pattern
            # fall back to single task case
            task_label = 0
        else:
            task_label = task_labels[0]

        kl_div, reconstruction_loss = strategy.loss
        loss = 0

        if self.with_kl_loss:
            loss -= kl_div

        if self.with_reconstruction_loss:
            loss -= reconstruction_loss

        self._loss.update(loss, patterns=len(strategy.mb_y), task_label=task_label)

    def __str__(self):
        if self.with_kl_loss and self.with_reconstruction_loss:
            return "test/loss_stream"

        elif self.with_reconstruction_loss:
            return "test/reconstruction_loss_stream"
        else:
            return "test/kl_loss_stream"


def vae_ft_loss_metrics(*, experience=False, stream=False) -> t.List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param experience: If True, will return a metric able to log
        the loss on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the loss averaged over the entire evaluation stream of experiences.

    :return: A list of plugin metrics.
    """

    metrics = []

    if experience:
        metrics.extend(
            [
                VAEExperienceLoss(),
                VAEExperienceLoss(with_kl_loss=False),
                VAEExperienceLoss(with_reconstruction_loss=False),
            ]
        )

    if stream:
        metrics.extend(
            [
                VAEStreamLoss(),
                VAEStreamLoss(with_kl_loss=False),
                VAEStreamLoss(with_reconstruction_loss=False),
            ]
        )

    return metrics


__all__ = [
    "VAEExperienceLoss",
    "StreamLoss",
    "vae_ft_loss_metrics",
]
