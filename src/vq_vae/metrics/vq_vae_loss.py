import typing as t

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.evaluation.metrics import ExperienceLoss, StreamLoss
from src.avalanche.strategies import NaivePytorchLightning


class VqVaeExperienceLoss(ExperienceLoss):
    def __init__(
        self,
        with_vq_loss: bool = False,
        with_reconstruction_loss: bool = False,
        with_lin_loss: bool = False,
        with_lin_acc: bool = False,
    ):
        super().__init__()

        self.with_vq_loss = with_vq_loss
        self.with_reconstruction_loss = with_reconstruction_loss
        self.with_lin_acc = with_lin_acc
        self.with_lin_loss = with_lin_loss

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

        vq_loss, reconstruction_loss, clf_loss, clf_acc, _ = strategy.loss
        loss = 0

        if self.with_vq_loss:
            loss += vq_loss

        if self.with_reconstruction_loss:
            loss += reconstruction_loss

        if self.with_lin_loss:
            loss -= clf_loss

        if self.with_lin_acc:
            loss += clf_acc

        self._loss.update(loss, patterns=len(strategy.mb_y), task_label=task_label)

    def __str__(self):
        if self.with_vq_loss and self.with_reconstruction_loss:
            return "test/loss_exp/full"

        elif self.with_reconstruction_loss:
            return "test/loss_exp/reconstruction"
        elif self.with_vq_loss:
            return "test/loss_exp/kl"
        elif self.with_lin_loss:
            return "test/loss_exp/clf_loss"
        elif self.with_lin_acc:
            return "test/loss_exp/clf_accuracy"

    def _package_result(self, strategy: "NaivePytorchLightning") -> "MetricResult":
        metric_value = self.result(strategy)
        add_exp = False
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = (
                    f"{self}/experience_step_{strategy.experience_step}/task_{k}"
                )
                metrics.append(MetricValue(self, metric_name, v, plot_x_position))
            return metrics
        else:
            metric_name = f"{self}/experience_step_{strategy.experience_step}"
            return [MetricValue(self, metric_name, metric_value, plot_x_position)]


class VqVaeStreamLoss(StreamLoss):
    def __init__(
        self,
        with_vq_loss: bool = False,
        with_reconstruction_loss: bool = False,
        with_lin_loss: bool = False,
        with_lin_acc: bool = False,
    ):
        super().__init__()

        self.with_vq_loss = with_vq_loss
        self.with_reconstruction_loss = with_reconstruction_loss
        self.with_lin_acc = with_lin_acc
        self.with_lin_loss = with_lin_loss

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

        vq_loss, reconstruction_loss, clf_loss, clf_acc, _ = strategy.loss
        loss = 0

        if self.with_vq_loss:
            loss += vq_loss

        if self.with_reconstruction_loss:
            loss += reconstruction_loss

        if self.with_lin_loss:
            loss -= clf_loss

        if self.with_lin_acc:
            loss += clf_acc

        self._loss.update(loss, patterns=len(strategy.mb_y), task_label=task_label)

    def __str__(self):
        if self.with_vq_loss and self.with_reconstruction_loss:
            return "test/loss_stream/full"
        elif self.with_reconstruction_loss:
            return "test/loss_stream/reconstruction"
        elif self.with_vq_loss:
            return "test/loss_stream/kl"
        elif self.with_lin_loss:
            return "test/loss_stream/clf_loss"
        elif self.with_lin_acc:
            return "test/loss_stream/clf_accuracy"

    def _package_result(self, strategy: "NaivePytorchLightning") -> "MetricResult":
        metric_value = self.result(strategy)
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = (
                    f"{self}/experience_step_{strategy.experience_step}/task_{k}"
                )
                metrics.append(MetricValue(self, metric_name, v, plot_x_position))
            return metrics
        else:
            metric_name = f"{self}/experience_step_{strategy.experience_step}"
            return [MetricValue(self, metric_name, metric_value, plot_x_position)]


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
                VqVaeExperienceLoss(with_reconstruction_loss=True, with_vq_loss=True),
                VqVaeExperienceLoss(with_vq_loss=True),
                VqVaeExperienceLoss(with_reconstruction_loss=True),
                VqVaeExperienceLoss(with_lin_loss=True),
                VqVaeExperienceLoss(with_lin_acc=True),
            ]
        )

    if stream:
        metrics.extend(
            [
                VqVaeStreamLoss(with_reconstruction_loss=True, with_vq_loss=True),
                VqVaeStreamLoss(with_vq_loss=True),
                VqVaeStreamLoss(with_reconstruction_loss=True),
                VqVaeStreamLoss(with_lin_loss=True),
                VqVaeStreamLoss(with_lin_acc=True),
            ]
        )

    return metrics
