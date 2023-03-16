import typing as t

from wandb.wandb_torch import torch

import wandb
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import (
    AlternativeValues,
    MetricResult,
    MetricValue,
)
from avalanche.evaluation.metric_utils import (
    default_cm_image_creator,
    phase_and_task,
    stream_type,
)
from avalanche.evaluation.metrics import (
    StreamConfusionMatrix,
    WandBStreamConfusionMatrix,
)
from src.avalanche.strategies import NaivePytorchLightning


class VQVaeWandBStreamConfusionMatrix(WandBStreamConfusionMatrix):
    def after_eval_iteration(self, strategy: "NaivePytorchLightning"):
        *_, logits = strategy.mb_output
        self.update(logits, strategy.mb_y)

    def __str__(self):
        return "test/confusion_matrix_stream"

    # def _package_result(self, strategy: "NaivePytorchLightning") -> MetricResult:
    #     outputs, targets = self.result()
    #     phase_name, _ = phase_and_task(strategy)
    #     stream = stream_type(strategy.experience)
    #     metric_name = f"{str(self)}/experience_step_{strategy.experience_step}"
    #     plot_x_position = strategy.clock.train_iterations
    #
    #     # compute predicted classes
    #     preds = torch.argmax(outputs, dim=1).cpu().numpy()
    #     result = wandb.plot.confusion_matrix(
    #         preds=preds,
    #         y_true=targets.cpu().numpy(),
    #         class_names=self.class_names,
    #     )
    #
    #     metric_representation = MetricValue(
    #         self, metric_name, AlternativeValues(result), plot_x_position
    #     )
    #
    #     return [metric_representation]


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
