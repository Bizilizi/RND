import typing as t

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_utils import default_cm_image_creator
from avalanche.evaluation.metrics import (
    StreamConfusionMatrix,
    WandBStreamConfusionMatrix,
)


class VQVaeWandBStreamConfusionMatrix(WandBStreamConfusionMatrix):
    def after_eval_iteration(self, strategy: "SupervisedTemplate"):
        _, _, module_downstream_pred = strategy.mb_output
        self.update(module_downstream_pred, strategy.mb_y)

    def __str__(self):
        return "test/confusion_matrix_stream"


class VQVaeStreamConfusionMatrix(StreamConfusionMatrix):
    def after_eval_iteration(self, strategy: "SupervisedTemplate") -> None:
        *_, logits = strategy.mb_output
        self.update(logits, strategy.mb_y)


def vq_vae_confusion_matrix_metrics(
    num_classes=None,
    normalize=None,
    save_image=True,
    image_creator=default_cm_image_creator,
    class_names=None,
    stream=False,
    wandb=False,
    absolute_class_order: bool = False,
) -> t.List[PluginMetric]:
    metrics = []

    if stream:
        metrics.append(
            VQVaeStreamConfusionMatrix(
                num_classes=num_classes,
                normalize=normalize,
                save_image=save_image,
                image_creator=image_creator,
                absolute_class_order=absolute_class_order,
            )
        )
        if wandb:
            metrics.append(VQVaeWandBStreamConfusionMatrix(class_names=class_names))

    return metrics
