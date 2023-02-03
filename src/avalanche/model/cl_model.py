import typing as t
import pytorch_lightning as pl
import torch
from avalanche.benchmarks import CLExperience, NCExperience


class CLModel(pl.LightningModule):
    experience_step: int
    experience: t.Union[CLExperience, NCExperience]

    @staticmethod
    def criterion(
        x: t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        y: t.Optional[torch.Tensor] = None,
    ) -> t.Any:
        raise NotImplementedError

    def log_with_postfix(self, name: str, value: t.Any, *args, **kwargs):
        self.log_dict(
            {
                f"{name}/experience_step_{self.experience_step}": value,
            },
            *args,
            **kwargs,
        )
