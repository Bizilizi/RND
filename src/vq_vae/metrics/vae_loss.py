import typing as t

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name
from avalanche.evaluation.metrics import ExperienceLoss, StreamLoss
from avalanche.training.templates import SupervisedTemplate


class VqVaeExperienceLoss(ExperienceLoss):
    def __init__(
        self, with_vq_loss: bool = True, with_reconstruction_loss: bool = True
    ):
        super().__init__()

        self.with_vq_loss = with_vq_loss
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

        vq_loss, reconstruction_loss, _ = strategy.loss
        loss = 0

        if self.with_vq_loss:
            loss += vq_loss

        if self.with_reconstruction_loss:
            loss += reconstruction_loss

        self._loss.update(loss, patterns=len(strategy.mb_y), task_label=task_label)

    def __str__(self):
        if self.with_vq_loss and self.with_reconstruction_loss:
            return "test/loss_exp/full"

        elif self.with_reconstruction_loss:
            return "test/loss_exp/reconstruction"
        else:
            return "test/loss_exp/kl"

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        metric_value = self.result(strategy)
        add_exp = False
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k
                )
                metrics.append(MetricValue(self, metric_name, v, plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(
                self, strategy, add_experience=add_exp, add_task=True
            )
            return [MetricValue(self, metric_name, metric_value, plot_x_position)]


class VqVaeStreamLoss(StreamLoss):
    def __init__(
        self, with_vq_loss: bool = True, with_reconstruction_loss: bool = True
    ):
        super().__init__()

        self.with_vq_loss = with_vq_loss
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

        vq_loss, reconstruction_loss, _ = strategy.loss
        loss = 0

        if self.with_vq_loss:
            loss += vq_loss

        if self.with_reconstruction_loss:
            loss += reconstruction_loss

        self._loss.update(loss, patterns=len(strategy.mb_y), task_label=task_label)

    def __str__(self):
        if self.with_vq_loss and self.with_reconstruction_loss:
            return "test/loss_stream/full"

        elif self.with_reconstruction_loss:
            return "test/loss_stream/reconstruction"
        else:
            return "test/loss_stream/kl"


def vq_vae_loss_metrics(*, experience=False, stream=False) -> t.List[PluginMetric]:
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
                VqVaeExperienceLoss(),
                VqVaeExperienceLoss(with_vq_loss=False),
                VqVaeExperienceLoss(with_reconstruction_loss=False),
            ]
        )

    if stream:
        metrics.extend(
            [
                VqVaeStreamLoss(),
                VqVaeStreamLoss(with_vq_loss=False),
                VqVaeStreamLoss(with_reconstruction_loss=False),
            ]
        )

    return metrics
