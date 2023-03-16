import typing as t

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.evaluation.metrics import (
    GenericExperienceForgetting,
    GenericStreamForgetting,
    Mean,
)
from src.avalanche.strategies import NaivePytorchLightning


class VqVaeStreamForgetting(GenericStreamForgetting):
    def __init__(
        self,
        with_vq_loss: bool = False,
        with_reconstruction_loss: bool = False,
        with_lin_loss: bool = False,
        with_lin_acc: bool = False,
    ):

        super().__init__()
        assert (
            with_vq_loss or with_reconstruction_loss or with_lin_loss or with_lin_acc
        ), "One of the parameters (with_rnd_loss, with_downstream_loss) have to present"

        self.with_vq_loss = with_vq_loss
        self.with_reconstruction_loss = with_reconstruction_loss
        self.with_lin_acc = with_lin_acc
        self.with_lin_loss = with_lin_loss

        self._current_metric = Mean()
        """
        The average over the current evaluation experience
        """

    def metric_update(self, strategy):
        vq_loss, reconstruction_loss, clf_loss, clf_acc, _ = strategy.loss
        loss = 0

        if self.with_vq_loss:
            loss -= vq_loss

        if self.with_reconstruction_loss:
            loss -= reconstruction_loss

        if self.with_lin_loss:
            loss -= clf_loss

        if self.with_lin_acc:
            loss += clf_acc

        self._current_metric.update(loss, 1)

    def metric_result(self, strategy):
        return self._current_metric.result()

    def __str__(self):
        if self.with_reconstruction_loss and self.with_vq_loss:
            return "test/forgetting_stream/full"
        elif self.with_reconstruction_loss:
            return "test/forgetting_stream/reconstruction"
        elif self.with_reconstruction_loss:
            return "test/forgetting_stream/vq"
        elif self.with_lin_loss:
            return "test/forgetting_stream/clf_loss"
        elif self.with_lin_acc:
            return "test/forgetting_stream/clf_accuracy"

        return "undefined"

    def _package_result(self, strategy: "NaivePytorchLightning") -> MetricResult:
        metric_value = self.result()
        metric_name = str(self)
        plot_x_position = strategy.clock.train_iterations

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]


class VqVaeExperienceForgetting(GenericExperienceForgetting):
    def __init__(
        self,
        with_vq_loss: bool = False,
        with_reconstruction_loss: bool = False,
        with_lin_loss: bool = False,
        with_lin_acc: bool = False,
    ):

        super().__init__()

        assert (
            with_vq_loss or with_reconstruction_loss or with_lin_loss or with_lin_acc
        ), "One of the parameters (with_rnd_loss, with_downstream_loss) have to present"

        self.with_vq_loss = with_vq_loss
        self.with_reconstruction_loss = with_reconstruction_loss
        self.with_lin_acc = with_lin_acc
        self.with_lin_loss = with_lin_loss

        self._current_metric = Mean()
        """
        The average over the current evaluation experience
        """

    def metric_update(self, strategy):
        vq_loss, reconstruction_loss, clf_loss, clf_acc, _ = strategy.loss
        loss = 0

        if self.with_vq_loss:
            loss -= vq_loss

        if self.with_reconstruction_loss:
            loss -= reconstruction_loss

        if self.with_lin_loss:
            loss -= clf_loss

        if self.with_lin_acc:
            loss += clf_acc

        self._current_metric.update(loss, 1)

    def metric_result(self, strategy):
        return self._current_metric.result()

    def __str__(self):

        if self.with_reconstruction_loss and self.with_vq_loss:
            return "test/forgetting_exp/full"
        elif self.with_reconstruction_loss:
            return "test/forgetting_exp/reconstruction"
        elif self.with_reconstruction_loss:
            return "test/forgetting_exp/vq"
        elif self.with_lin_loss:
            return "test/forgetting_exp/clf_loss"
        elif self.with_lin_acc:
            return "test/forgetting_exp/clf_accuracy"

        return "undefined"

    def _package_result(self, strategy: "NaivePytorchLightning") -> MetricResult:
        # this checks if the evaluation experience has been
        # already encountered at training time
        # before the last training.
        # If not, forgetting should not be returned.
        forgetting = self.result(k=self.eval_exp_id)
        if forgetting is not None:
            metric_name = f"{self}/experience_step_{strategy.experience_step}"
            plot_x_position = strategy.clock.train_iterations

            metric_values = [
                MetricValue(self, metric_name, forgetting, plot_x_position)
            ]
            return metric_values


def vq_vae_forgetting_metrics(
    *, experience=False, stream=False
) -> t.List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param experience: If True, will return a metric able to log
        the forgetting on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the forgetting averaged over the evaluation stream experiences,
        which have been observed during training.

    :return: A list of plugin metrics.
    """

    metrics = []

    if experience:
        metrics.extend(
            [
                VqVaeExperienceForgetting(
                    with_reconstruction_loss=True, with_vq_loss=True
                ),
                VqVaeExperienceForgetting(with_reconstruction_loss=True),
                VqVaeExperienceForgetting(with_vq_loss=True),
                VqVaeExperienceForgetting(with_lin_loss=True),
                VqVaeExperienceForgetting(with_lin_acc=True),
            ]
        )

    if stream:
        metrics.extend(
            [
                VqVaeStreamForgetting(with_reconstruction_loss=True, with_vq_loss=True),
                VqVaeStreamForgetting(with_reconstruction_loss=True),
                VqVaeStreamForgetting(with_vq_loss=True),
                VqVaeStreamForgetting(with_lin_loss=True),
                VqVaeStreamForgetting(with_lin_acc=True),
            ]
        )

    return metrics
