import typing as t

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics import ExperienceLoss, StreamLoss


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
        criterion_output = strategy.loss
        loss = 0

        if self.with_vq_loss:
            loss += criterion_output.vq_loss

        if self.with_reconstruction_loss:
            loss += criterion_output.reconstruction_loss

        if self.with_lin_loss:
            loss -= criterion_output.clf_loss

        if self.with_lin_acc:
            loss += criterion_output.clf_acc

        if self.split_by_task:
            self._loss.update(
                loss=loss,
                patterns=len(strategy.mb_y),
                task_label=strategy.mb_task_id,
            )
        else:
            self._loss.update(loss, patterns=len(strategy.mb_y))

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
        criterion_output = strategy.loss
        loss = 0

        if self.with_vq_loss:
            loss += criterion_output.vq_loss

        if self.with_reconstruction_loss:
            loss += criterion_output.reconstruction_loss

        if self.with_lin_loss:
            loss -= criterion_output.clf_loss

        if self.with_lin_acc:
            loss += criterion_output.clf_acc

        if self.split_by_task:
            self._loss.update(
                loss=loss,
                patterns=len(strategy.mb_y),
                task_label=strategy.mb_task_id,
            )
        else:
            self._loss.update(loss, patterns=len(strategy.mb_y))

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
