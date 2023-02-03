import typing as t

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics import (
    GenericExperienceForgetting,
    GenericStreamForgetting,
    Mean,
)


class VAEStreamForgetting(GenericStreamForgetting):
    def __init__(
        self, with_kl_loss: bool = True, with_reconstruction_loss: bool = True
    ):

        super().__init__()

        assert (
            with_kl_loss or with_reconstruction_loss
        ), "One of the parameters (with_rnd_loss, with_downstream_loss) have to present"

        self.with_kl_loss = with_kl_loss
        self.with_reconstruction_loss = with_reconstruction_loss
        self._current_metric = Mean()
        """
        The average over the current evaluation experience
        """

    def metric_update(self, strategy):
        kl_div, reconstruction_loss = strategy.loss
        loss = 0

        if self.with_kl_loss:
            loss -= kl_div

        if self.with_reconstruction_loss:
            loss -= reconstruction_loss

        self._current_metric.update(loss, 1)

    def metric_result(self, strategy):
        return self._current_metric.result()

    def __str__(self):
        name = "StreamForgetting"

        if self.with_reconstruction_loss:
            name += "/REC"

        if self.with_kl_loss:
            name += "/KL"

        return name


class RNDExperienceForgetting(GenericExperienceForgetting):
    def __init__(
        self, with_kl_loss: bool = True, with_reconstruction_loss: bool = True
    ):

        super().__init__()

        assert (
            with_kl_loss or with_reconstruction_loss
        ), "One of the parameters (with_rnd_loss, with_downstream_loss) have to present"

        self.with_kl_loss = with_kl_loss
        self.with_reconstruction_loss = with_reconstruction_loss
        self._current_metric = Mean()
        """
        The average over the current evaluation experience
        """

    def metric_update(self, strategy):
        kl_div, reconstruction_loss = strategy.loss
        loss = 0

        if self.with_kl_loss:
            loss -= kl_div

        if self.with_reconstruction_loss:
            loss -= reconstruction_loss

        self._current_metric.update(loss, 1)

    def metric_result(self, strategy):
        return self._current_metric.result()

    def __str__(self):
        name = "ExperienceForgetting"

        if self.with_reconstruction_loss:
            name += "/REC"

        if self.with_kl_loss:
            name += "/KL"

        return name


def vae_ft_forgetting_metrics(
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
                RNDExperienceForgetting(),
                RNDExperienceForgetting(with_reconstruction_loss=False),
                RNDExperienceForgetting(with_kl_loss=False),
            ]
        )

    if stream:
        metrics.extend(
            [
                VAEStreamForgetting(),
                VAEStreamForgetting(with_reconstruction_loss=False),
                VAEStreamForgetting(with_kl_loss=False),
            ]
        )

    return metrics
