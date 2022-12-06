from typing import List

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics.accuracy import AccuracyPluginMetric


class ExperienceAccuracy(AccuracyPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceAccuracy, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def update(self, strategy):
        # task labels defined for each experience
        if hasattr(strategy.experience, "task_labels"):
            task_labels = strategy.experience.task_labels
        else:
            task_labels = [0]  # add fixed task label if not available.

        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]

        _, _, module_downstream_pred = strategy.mb_output
        self._accuracy.update(module_downstream_pred, strategy.mb_y, task_labels)

    def __str__(self):
        return "Top1_Acc_Exp"


class StreamAccuracy(AccuracyPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average accuracy over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamAccuracy metric
        """
        super(StreamAccuracy, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def update(self, strategy):
        # task labels defined for each experience
        if hasattr(strategy.experience, "task_labels"):
            task_labels = strategy.experience.task_labels
        else:
            task_labels = [0]  # add fixed task label if not available.

        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]

        _, _, module_downstream_pred = strategy.mb_output
        self._accuracy.update(module_downstream_pred, strategy.mb_y, task_labels)

    def __str__(self):
        return "Top1_Acc_Stream"


def rnd_accuracy_metrics(
    *,
    experience=False,
    stream=False,
) -> List[PluginMetric]:
    """
    :param experience: If True, will return a metric able to log
        the accuracy on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the accuracy averaged over the entire evaluation stream of experiences.

    :return: A list of plugin metrics.
    """

    metrics = []

    if experience:
        metrics.append(ExperienceAccuracy())

    if stream:
        metrics.append(StreamAccuracy())

    return metrics


__all__ = [
    "ExperienceAccuracy",
    "StreamAccuracy",
    "rnd_accuracy_metrics",
]
