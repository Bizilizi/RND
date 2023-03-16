import typing as t
from functools import partial
from itertools import chain

from avalanche.benchmarks import (
    AvalancheSubset,
    NCExperience,
    ClassificationStream,
    NCScenario,
    GenericClassificationExperience,
)


def shift_class(y, current_experience, num_tasks_in_batch):
    """named function so to make the class compatible with pickle"""
    return y + current_experience * num_tasks_in_batch


class NCExperienceWithOriginClasses(
    GenericClassificationExperience[
        NCScenario, ClassificationStream["NCExperience", NCScenario]
    ]
):
    """
    Defines a "New Classes" experience. It defines fields to obtain the current
    dataset and the associated task label. It also keeps a reference to the
    stream from which this experience was taken.
    """

    def __init__(
        self,
        origin_stream: ClassificationStream["NCExperience", NCScenario],
        current_experience: int,
        num_tasks_in_batch: int,
    ):
        """
        Creates a ``NCExperience`` instance given the stream from this
        experience was taken and the current experience ID.

        :param origin_stream: The stream from which this experience was
            obtained.
        :param current_experience: The current experience ID, as an integer.
        """
        super().__init__(origin_stream, current_experience)

        self.num_tasks_in_batch = num_tasks_in_batch
        self.dataset = self.dataset.add_transforms(
            target_transform=partial(
                shift_class,
                num_tasks_in_batch=num_tasks_in_batch,
                current_experience=current_experience,
            )
        )


def shift_experiences_classes(benchmark, num_tasks_in_batch: int):
    """
    Shifts all the returned classes in experience dataset by task_id * num_tasks_in_batch
    """
    benchmark.experience_factory = partial(
        NCExperienceWithOriginClasses, num_tasks_in_batch=num_tasks_in_batch
    )

    return benchmark
