import errno
import os
import sys
import typing as t
from pathlib import Path

import numpy as np
import torch
from matplotlib.pyplot import Figure
from numpy import array
from PIL.Image import Image
from torch import Tensor

from avalanche.benchmarks import OnlineCLExperience
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metric_results import (
    AlternativeValues,
    MetricValue,
    TensorImage,
)
from avalanche.evaluation.metric_utils import phase_and_task, stream_type
from avalanche.logging import BaseLogger

if t.TYPE_CHECKING:
    from avalanche.evaluation.metric_results import MetricValue
    from avalanche.training.templates.supervised import SupervisedTemplate

from tqdm import tqdm

from wandb import Image


class InteractiveWandBLogger(BaseLogger, SupervisedPlugin):
    def __init__(
        self,
        project_name: str = "Avalanche",
        run_name: str = "Test",
        log_artifacts: bool = False,
        path: t.Union[str, Path] = "Checkpoints",
        uri: str = None,
        sync_tfboard: bool = False,
        save_code: bool = True,
        config: object = None,
        dir: t.Union[str, Path] = None,
        params: dict = None,
    ):
        """Creates an instance of the `WandBLogger`.

        :param project_name: Name of the W&B project.
        :param run_name: Name of the W&B run.
        :param log_artifacts: Option to log model weights as W&B Artifacts.
        :param path: Path to locally save the model checkpoints.
        :param uri: URI identifier for external storage buckets (GCS, S3).
        :param sync_tfboard: Syncs TensorBoard to the W&B dashboard UI.
        :param save_code: Saves the main training script to W&B.
        :param config: Syncs hyper-parameters and config values used to W&B.
        :param dir: Path to the local log directory for W&B logs to be saved at.
        :param params: All arguments for wandb.init() function call. Visit
            https://docs.wandb.ai/ref/python/init to learn about all
            wand.init() parameters.
        """
        super().__init__()
        self.import_wandb()
        self.project_name = project_name
        self.run_name = run_name
        self.log_artifacts = log_artifacts
        self.path = path
        self.uri = uri
        self.sync_tfboard = sync_tfboard
        self.save_code = save_code
        self.config = config
        self.dir = dir
        self.params = params
        self.args_parse()
        self.before_run()
        self.step = 0
        self.exp_count = 0
        self.file = sys.stdout
        self._pbar = None
        self.last_length = 0

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError('Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    def args_parse(self):
        self.init_kwargs = {
            "project": self.project_name,
            "name": self.run_name,
            "sync_tensorboard": self.sync_tfboard,
            "dir": self.dir,
            "save_code": self.save_code,
            "config": self.config,
        }
        if self.params:
            self.init_kwargs.update(self.params)

    def before_run(self):
        if self.wandb is None:
            self.import_wandb()

        if self.wandb.run is not None:
            if self.init_kwargs:
                self.wandb.init(**self.init_kwargs)
            else:
                self.wandb.init()

        self.wandb.run.define_metric("avalanche/TrainingExperience")
        self.wandb.run.define_metric("avalanche/MetricStep")

    def after_training_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: t.List["MetricValue"],
        **kwargs: t.Dict[str, t.Any],
    ):
        if isinstance(strategy.experience, OnlineCLExperience):
            experience = strategy.experience.logging()
            if experience.is_last_subexp:
                self._end_progress()
                super().after_training_exp(strategy, metric_values, **kwargs)

        for val in metric_values:
            self.log_metrics([val])

        self.wandb.log(
            {
                "avalanche/TrainingExperience": self.exp_count,
                "avalanche/MetricStep": self.step,
            }
        )
        self.exp_count += 1

    def log_single_metric(self, name, value, x_plot):
        self.step = x_plot

        if isinstance(value, AlternativeValues):
            value = value.best_supported_value(
                Image,
                Tensor,
                TensorImage,
                Figure,
                float,
                int,
                self.wandb.viz.CustomChart,
            )

        if not isinstance(
            value,
            (Image, Tensor, Figure, float, int, self.wandb.viz.CustomChart),
        ):
            # Unsupported type
            return

        if isinstance(value, Image):
            self.wandb.log(
                {
                    name: self.wandb.Image(value),
                    "avalanche/MetricStep": self.step,
                    "avalanche/TrainingExperience": self.exp_count,
                }
            )

        elif isinstance(value, Tensor):
            value = np.histogram(value.view(-1).numpy())
            self.wandb.log(
                {
                    name: self.wandb.Histogram(np_histogram=value),
                    "avalanche/MetricStep": self.step,
                    "avalanche/TrainingExperience": self.exp_count,
                },
            )

        elif isinstance(value, (float, int, Figure, self.wandb.viz.CustomChart)):
            self.wandb.log(
                {
                    name: value,
                    "avalanche/MetricStep": self.step,
                    "avalanche/TrainingExperience": self.exp_count,
                }
            )

        elif isinstance(value, TensorImage):
            self.wandb.log(
                {
                    name: self.wandb.Image(array(value)),
                    "avalanche/MetricStep": self.step,
                    "avalanche/TrainingExperience": self.exp_count,
                }
            )

        elif name.startswith("WeightCheckpoint"):
            if self.log_artifacts:
                cwd = os.getcwd()
                ckpt = os.path.join(cwd, self.path)
                try:
                    os.makedirs(ckpt)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                suffix = ".pth"
                dir_name = os.path.join(ckpt, name + suffix)
                artifact_name = os.path.join("Models", name + suffix)
                if isinstance(value, Tensor):
                    torch.save(value, dir_name)
                    name = os.path.splittext(self.checkpoint)
                    artifact = self.wandb.Artifact(name, type="model")
                    artifact.add_file(dir_name, name=artifact_name)
                    self.wandb.run.log_artifact(artifact)
                    if self.uri is not None:
                        artifact.add_reference(self.uri, name=artifact_name)

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

    # region Validation
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

    # endregion


__all__ = ["InteractiveWandBLogger"]
