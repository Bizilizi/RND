import sys
import typing as t

from avalanche.benchmarks import OnlineCLExperience
from avalanche.logging import InteractiveLogger, WandBLogger
from tqdm import tqdm


class InteractiveWandBLogger(WandBLogger):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._pbar = None
        self.last_length = 0

    def before_training_epoch(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        super().before_training_epoch(strategy, metric_values, **kwargs)
        self._progress.total = len(strategy.dataloader)

    def after_training_epoch(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        self._end_progress()
        super().after_training_epoch(strategy, metric_values, **kwargs)

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        if isinstance(strategy.experience, OnlineCLExperience):
            experience = strategy.experience.logging()
            if experience.is_first_subexp:
                super().before_training_exp(strategy, metric_values, **kwargs)
                self._progress.total = (
                    experience.sub_stream_length
                    * strategy.train_passes
                    * (experience.subexp_size // strategy.train_mb_size)
                )
                self.last_length = self._progress.total

    def after_training_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        if isinstance(strategy.experience, OnlineCLExperience):
            experience = strategy.experience.logging()
            if experience.is_last_subexp:
                self._end_progress()
                super().after_training_exp(strategy, metric_values, **kwargs)

        super(WandBLogger, self).after_training_exp(strategy, metric_values, **kwargs)

    def before_eval_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        super().before_eval_exp(strategy, metric_values, **kwargs)
        self._progress.total = len(strategy.dataloader)

    def after_eval_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        self._end_progress()
        super().after_eval_exp(strategy, metric_values, **kwargs)

    def after_training_iteration(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        self._progress.update()
        self._progress.refresh()
        super().after_training_iteration(strategy, metric_values, **kwargs)

    def after_eval_iteration(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        self._progress.update()
        self._progress.refresh()
        super().after_eval_iteration(strategy, metric_values, **kwargs)

    @property
    def _progress(self):
        if self._pbar is None:
            self._pbar = tqdm(leave=True, position=0, file=sys.stdout)
        return self._pbar

    def _end_progress(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


__all__ = ["InteractiveWandBLogger"]
