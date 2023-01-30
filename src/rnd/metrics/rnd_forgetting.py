import typing as t

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics import (
    ExperienceForgetting,
    GenericExperienceForgetting,
    GenericStreamForgetting,
    Mean,
    StreamForgetting,
)


class RNDStreamAccForgetting(StreamForgetting):
    def metric_update(self, strategy):
        _, _, module_downstream_pred = strategy.mb_output
        self._current_metric.update(strategy.mb_y, module_downstream_pred, 0)


class RNDExperienceAccForgetting(ExperienceForgetting):
    def metric_update(self, strategy):
        _, _, module_downstream_pred = strategy.mb_output
        self._current_metric.update(strategy.mb_y, module_downstream_pred, 0)


class RNDStreamForgetting(GenericStreamForgetting):
    def __init__(self, with_rnd_loss: bool = True, with_downstream_loss: bool = True):

        super().__init__()

        assert (
            with_rnd_loss or with_downstream_loss
        ), "One of the parameters (with_rnd_loss, with_downstream_loss) have to present"

        self.with_rnd_loss = with_rnd_loss
        self.with_downstream_loss = with_downstream_loss
        self._current_metric = Mean()
        """
        The average over the current evaluation experience
        """

    def metric_update(self, strategy):
        rn_loss, downstream_loss = strategy.loss
        loss = 0

        if self.with_rnd_loss:
            loss -= rn_loss

        if self.with_downstream_loss:
            loss -= downstream_loss

        self._current_metric.update(loss, 1)

    def metric_result(self, strategy):
        return self._current_metric.result()

    def __str__(self):
        name = "StreamForgetting"

        if self.with_downstream_loss:
            name += "/DS"

        if self.with_rnd_loss:
            name += "/RN"

        return name


class RNDExperienceForgetting(GenericExperienceForgetting):
    def __init__(self, with_rnd_loss: bool = True, with_downstream_loss: bool = True):

        super().__init__()

        assert (
            with_rnd_loss or with_downstream_loss
        ), "One of the parameters (with_rnd_loss, with_downstream_loss) have to present"

        self.with_rnd_loss = with_rnd_loss
        self.with_downstream_loss = with_downstream_loss
        self._current_metric = Mean()
        """
        The average over the current evaluation experience
        """

    def metric_update(self, strategy):
        rn_loss, downstream_loss = strategy.loss
        loss = 0

        if self.with_rnd_loss:
            loss -= rn_loss

        if self.with_downstream_loss:
            loss -= downstream_loss

        self._current_metric.update(loss, 1)

    def metric_result(self, strategy):
        return self._current_metric.result()

    def __str__(self):
        name = "ExperienceForgetting"

        if self.with_downstream_loss:
            name += "/DS"

        if self.with_rnd_loss:
            name += "/RN"

        return name


def rnd_forgetting_metrics(
    *, experience=False, stream=False, accuracy=False
) -> t.List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param experience: If True, will return a metric able to log
        the forgetting on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the forgetting averaged over the evaluation stream experiences,
        which have been observed during training.
    :param accuracy: If True, will return a accuracy metric able to log
        the forgetting metrics, both for experience and stream.

    :return: A list of plugin metrics.
    """

    metrics = []

    if experience:
        metrics.extend(
            [
                RNDExperienceForgetting(),
                RNDExperienceForgetting(with_downstream_loss=False),
                RNDExperienceForgetting(with_rnd_loss=False),
            ]
        )

        if accuracy:
            metrics.append(RNDExperienceAccForgetting())

    if stream:
        metrics.extend(
            [
                RNDStreamForgetting(),
                RNDStreamForgetting(with_downstream_loss=False),
                RNDStreamForgetting(with_rnd_loss=False),
            ]
        )
        if accuracy:
            metrics.append(RNDStreamAccForgetting())

    return metrics
