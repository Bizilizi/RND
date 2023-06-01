from contextlib import contextmanager
from typing import Generator

from pytorch_lightning.plugins.precision import MixedPrecisionPlugin


class CustomMixedPrecisionPlugin(MixedPrecisionPlugin):
    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        yield
