import typing as t

import torch


def get_resume_arguments(resume_from: t.Optional[str]):
    if resume_from:
        return torch.load(resume_from)

    return None
