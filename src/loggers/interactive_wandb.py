import sys
import typing as t

from avalanche.benchmarks import OnlineCLExperience
from avalanche.evaluation.metric_utils import phase_and_task, stream_type
from avalanche.logging import WandBLogger
from tqdm import tqdm


class InteractiveWandBLogger(WandBLogger):
    def __init__(
        self,
        file=sys.stdout,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.file = file
        self._pbar = None
        self.last_length = 0

    # region Training
    def before_training(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        super().before_training(strategy, metric_values, **kwargs)
        print("-- >> Start of training phase << --", file=self.file, flush=True)

    def before_training_epoch(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        super().before_training_epoch(strategy, metric_values, **kwargs)
        self._progress.total = len(strategy.dataloader)

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        self._on_exp_start(strategy)
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

    def after_training_iteration(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        self._progress.update()
        self._progress.refresh()
        super().after_training_iteration(strategy, metric_values, **kwargs)

    def after_training_epoch(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        super().after_training_epoch(strategy, metric_values, **kwargs)
        print(
            f"Epoch {strategy.clock.train_exp_epochs} ended.",
            file=self.file,
            flush=True,
        )
        self.print_current_metrics()
        self.metric_vals = {}
        self._end_progress()

    def after_training(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        super().after_training(strategy, metric_values, **kwargs)
        print("-- >> End of training phase << --", file=self.file, flush=True)

    # endregion

    def before_eval(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        super().before_eval(strategy, metric_values, **kwargs)
        print("-- >> Start of eval phase << --", file=self.file, flush=True)

    def before_eval_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        self._on_exp_start(strategy)
        super().before_eval_exp(strategy, metric_values, **kwargs)
        self._progress.total = len(strategy.dataloader)

    def after_eval_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        super().after_eval_exp(strategy, metric_values, **kwargs)
        exp_id = strategy.experience.current_experience
        task_id = phase_and_task(strategy)[1]
        if task_id is None:
            print(
                f"> Eval on experience {exp_id} "
                f"from {stream_type(strategy.experience)} stream ended.",
                file=self.file,
                flush=True,
            )
        else:
            print(
                f"> Eval on experience {exp_id} (Task "
                f"{task_id}) "
                f"from {stream_type(strategy.experience)} stream ended.",
                file=self.file,
                flush=True,
            )
        self.print_current_metrics()
        self.metric_vals = {}
        self._end_progress()

    def after_eval_iteration(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        self._progress.update()
        self._progress.refresh()
        super().after_eval_iteration(strategy, metric_values, **kwargs)

    def after_eval(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs,
    ):
        super().after_eval(strategy, metric_values, **kwargs)
        print("-- >> End of eval phase << --", file=self.file, flush=True)
        self.print_current_metrics()
        self.metric_vals = {}

    @property
    def _progress(self):
        if self._pbar is None:
            self._pbar = tqdm(leave=True, position=0, file=sys.stdout)
        return self._pbar

    def _end_progress(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def _on_exp_start(self, strategy: "SupervisedTemplate"):
        action_name = "training" if strategy.is_training else "eval"
        exp_id = strategy.experience.current_experience
        task_id = phase_and_task(strategy)[1]
        stream = stream_type(strategy.experience)
        if task_id is None:
            print(
                "-- Starting {} on experience {} from {} stream --".format(
                    action_name, exp_id, stream
                ),
                file=self.file,
                flush=True,
            )
        else:
            print(
                "-- Starting {} on experience {} (Task {}) from {}"
                " stream --".format(action_name, exp_id, task_id, stream),
                file=self.file,
                flush=True,
            )


__all__ = ["InteractiveWandBLogger"]
