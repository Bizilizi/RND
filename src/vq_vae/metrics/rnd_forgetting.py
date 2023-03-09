import typing as t

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.evaluation.metric_utils import get_metric_name
from avalanche.evaluation.metrics import (
    GenericExperienceForgetting,
    GenericStreamForgetting,
    Mean,
)


class VqVaeStreamForgetting(GenericStreamForgetting):
    def __init__(
        self, with_vq_loss: bool = True, with_reconstruction_loss: bool = True
    ):

        super().__init__()

        assert (
            with_vq_loss or with_reconstruction_loss
        ), "One of the parameters (with_rnd_loss, with_downstream_loss) have to present"

        self.with_vq_loss = with_vq_loss
        self.with_reconstruction_loss = with_reconstruction_loss
        self._current_metric = Mean()
        """
        The average over the current evaluation experience
        """

    def metric_update(self, strategy):
        vq_loss, reconstruction_loss, _ = strategy.loss
        loss = 0

        if self.with_vq_loss:
            loss -= vq_loss

        if self.with_reconstruction_loss:
            loss -= reconstruction_loss

        self._current_metric.update(loss, 1)

    def metric_result(self, strategy):
        return self._current_metric.result()

    def __str__(self):
        if self.with_reconstruction_loss and self.with_vq_loss:
            return "test/forgetting_stream/full"
        elif self.with_reconstruction_loss:
            return "test/forgetting_stream/reconstruction"
        else:
            return "test/forgetting_stream/vq"


class VqVaeExperienceForgetting(GenericExperienceForgetting):
    def __init__(
        self, with_vq_loss: bool = True, with_reconstruction_loss: bool = True
    ):

        super().__init__()

        assert (
            with_vq_loss or with_reconstruction_loss
        ), "One of the parameters (with_rnd_loss, with_downstream_loss) have to present"

        self.with_vq_loss = with_vq_loss
        self.with_reconstruction_loss = with_reconstruction_loss
        self._current_metric = Mean()
        """
        The average over the current evaluation experience
        """

    def metric_update(self, strategy):
        vq_loss, reconstruction_loss, _ = strategy.loss
        loss = 0

        if self.with_vq_loss:
            loss -= vq_loss

        if self.with_reconstruction_loss:
            loss -= reconstruction_loss

        self._current_metric.update(loss, 1)

    def metric_result(self, strategy):
        return self._current_metric.result()

    def __str__(self):

        if self.with_reconstruction_loss and self.with_vq_loss:
            return "test/forgetting_exp/full"
        elif self.with_reconstruction_loss:
            return "test/forgetting_exp/reconstruction"
        else:
            return "test/forgetting_exp/vq"

    def _package_result(self, strategy: "SupervisedTemplate") -> MetricResult:
        # this checks if the evaluation experience has been
        # already encountered at training time
        # before the last training.
        # If not, forgetting should not be returned.
        forgetting = self.result(k=self.eval_exp_id)
        if forgetting is not None:
            metric_name = get_metric_name(self, strategy)
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
                VqVaeExperienceForgetting(),
                VqVaeExperienceForgetting(with_reconstruction_loss=False),
                VqVaeExperienceForgetting(with_vq_loss=False),
            ]
        )

    if stream:
        metrics.extend(
            [
                VqVaeStreamForgetting(),
                VqVaeStreamForgetting(with_reconstruction_loss=False),
                VqVaeStreamForgetting(with_vq_loss=False),
            ]
        )

    return metrics
