import typing as t

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_utils import default_cm_image_creator
from avalanche.evaluation.metrics import (
    StreamConfusionMatrix,
    WandBStreamConfusionMatrix,
)


class RNDWandBStreamConfusionMatrix(WandBStreamConfusionMatrix):
    def after_eval_iteration(self, strategy: "SupervisedTemplate"):
        super(WandBStreamConfusionMatrix, self).after_eval_iteration(strategy)
        _, _, module_downstream_pred = strategy.mb_output
        self.update(module_downstream_pred, strategy.mb_y)


class RNDStreamConfusionMatrix(StreamConfusionMatrix):
    def after_eval_iteration(self, strategy: "SupervisedTemplate") -> None:
        super().after_eval_iteration(strategy)
        _, _, module_downstream_pred = strategy.mb_output
        self.update(module_downstream_pred, strategy.mb_y)


def rnd_confusion_matrix_metrics(
    num_classes=None,
    normalize=None,
    save_image=True,
    image_creator=default_cm_image_creator,
    class_names=None,
    stream=False,
    wandb=False,
    absolute_class_order: bool = False,
) -> t.List[PluginMetric]:
    """Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param num_classes: The number of classes. Defaults to None,
        which means that the number of classes will be inferred from
        ground truth and prediction Tensors (see class description for more
        details). If not None, the confusion matrix will always be of size
        `num_classes, num_classes` and only the first `num_classes` values
        of output logits or target logits will be considered in the update.
        If the output or targets are provided as numerical labels,
        there can be no label greater than `num_classes`.
    :param normalize: Normalizes confusion matrix over the true (rows),
        predicted (columns) conditions or all the population. If None,
        confusion matrix will not be normalized. Valid values are: 'true',
        'pred' and 'all' or None.
    :param save_image: If True, a graphical representation of the confusion
        matrix will be logged, too. If False, only the Tensor representation
        will be logged. Defaults to True.
    :param image_creator: A callable that, given the tensor representation
        of the confusion matrix, returns a graphical representation of the
        matrix as a PIL Image. Defaults to `default_cm_image_creator`.
    :param class_names: W&B only. List of names for the classes.
        E.g. ["cat", "dog"] if class 0 == "cat" and class 1 == "dog"
        If None, no class names will be used. Default None.
    :param stream: If True, will return a metric able to log
        the confusion matrix averaged over the entire evaluation stream
        of experiences.
    :param wandb: if True, will return a Weights and Biases confusion matrix
        together with all the other confusion matrixes requested.
    :param absolute_class_order: Not W&B. If true, the labels in the created
        image will be sorted by id, otherwise they will be sorted by order of
        encounter at training time. This parameter is ignored if `save_image` is
        False, or the scenario is not a NCScenario.

    :return: A list of plugin metrics.
    """

    metrics = []

    if stream:
        metrics.append(
            RNDStreamConfusionMatrix(
                num_classes=num_classes,
                normalize=normalize,
                save_image=save_image,
                image_creator=image_creator,
                absolute_class_order=absolute_class_order,
            )
        )
        if wandb:
            metrics.append(RNDWandBStreamConfusionMatrix(class_names=class_names))

    return metrics


__all__ = [
    "RNDStreamConfusionMatrix",
    "RNDWandBStreamConfusionMatrix",
    "rnd_confusion_matrix_metrics",
]
