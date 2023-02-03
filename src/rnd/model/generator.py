import typing as t

import torch


class ImageGenerator:
    def generate(
        self, num_samples: int, device: t.Optional[torch.device] = None
    ) -> torch.Tensor:
        ...
