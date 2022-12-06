import typing as t

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics import ExperienceLoss, StreamLoss


class RNDExperienceLoss(ExperienceLoss):
    def __init__(self, with_rnd_loss: bool = True, with_downstream_loss: bool = True):
        super().__init__()

        self.with_rnd_loss = with_rnd_loss
        self.with_downstream_loss = with_downstream_loss

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

        rn_loss, downstream_loss = strategy.loss
        loss = 0

        if self.with_rnd_loss:
            loss -= rn_loss

        if self.with_downstream_loss:
            loss -= downstream_loss

        self._loss.update(loss, patterns=len(strategy.mb_y), task_label=task_label)

    def __str__(self):
        if self.with_rnd_loss and self.with_downstream_loss:
            return "test/loss_exp"

        elif self.with_downstream_loss:
            return "test/downstream_loss_exp"
        else:
            return "test/rnd_loss_exp"


class RNDStreamLoss(StreamLoss):
    def __init__(self, with_rnd_loss: bool = True, with_downstream_loss: bool = True):
        super().__init__()

        self.with_rnd_loss = with_rnd_loss
        self.with_downstream_loss = with_downstream_loss

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

        rn_loss, downstream_loss = strategy.loss
        loss = 0

        if self.with_rnd_loss:
            loss -= rn_loss

        if self.with_downstream_loss:
            loss -= downstream_loss

        self._loss.update(loss, patterns=len(strategy.mb_y), task_label=task_label)

    def __str__(self):
        if self.with_rnd_loss and self.with_downstream_loss:
            return "test/loss_stream"

        elif self.with_downstream_loss:
            return "test/downstream_loss_stream"
        else:
            return "test/rnd_loss_stream"


def rnd_loss_metrics(*, experience=False, stream=False) -> t.List[PluginMetric]:
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
                RNDExperienceLoss(),
                RNDExperienceLoss(with_rnd_loss=False),
                RNDExperienceLoss(with_downstream_loss=False),
            ]
        )

    if stream:
        metrics.extend(
            [
                RNDStreamLoss(),
                RNDStreamLoss(with_rnd_loss=False),
                RNDStreamLoss(with_downstream_loss=False),
            ]
        )

    return metrics


__all__ = [
    "RNDExperienceLoss",
    "StreamLoss",
    "rnd_loss_metrics",
]
