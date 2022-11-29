import typing as t

import torch


class ImageGenerator(t.Protocol):
    def generate(self, num_samples: int, device: t.Optional[torch.device] = None):
        ...
