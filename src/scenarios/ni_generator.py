import typing as t

import torch
from avalanche.benchmarks import NIScenario, ni_benchmark
from avalanche.benchmarks.utils import SupportedDataset, concat_datasets_sequentially
from torch.distributions import Normal


def ni_benchmark_rand_normal(
    train_dataset: t.Union[t.Sequence[SupportedDataset], SupportedDataset],
    test_dataset: t.Union[t.Sequence[SupportedDataset], SupportedDataset],
    *,
    sigma_range_constant: int = 4,
    task_labels: bool = False,
    shuffle: bool = True,
    seed: t.Optional[int] = None,
    min_class_patterns_in_exp: int = 0,
    train_transform=None,
    eval_transform=None,
    reproducibility_data: t.Optional[t.Dict[str, t.Any]] = None,
) -> NIScenario:
    """This benchmark generator wraps Avalanche ni_benchmark
    however it creates a distribution of class samples in experiments
    similar to normal pdf drawn around class index
    """
    seq_train_dataset, seq_test_dataset = train_dataset, test_dataset
    if isinstance(train_dataset, list) or isinstance(train_dataset, tuple):
        if len(train_dataset) != len(test_dataset):
            raise ValueError(
                "Train/test dataset lists must contain the "
                "exact same number of datasets"
            )

        seq_train_dataset, seq_test_dataset, _ = concat_datasets_sequentially(
            train_dataset, test_dataset
        )

    if seed is not None:
        torch.random.manual_seed(seed)

    # First get number of classes and number of instances in dataset
    # for particular class
    unique_targets, unique_count = torch.unique(
        torch.as_tensor(seq_train_dataset.targets), return_counts=True
    )
    n_classes = len(unique_targets)

    # Get the patterns indexes for each class
    targets_as_tensor = torch.as_tensor(seq_train_dataset.targets)
    classes_to_patterns_idx = [
        torch.nonzero(torch.eq(targets_as_tensor, class_id)).view(-1).tolist()
        for class_id in range(n_classes)
    ]
    if shuffle:
        classes_to_patterns_idx = [
            torch.as_tensor(cls_patterns)[torch.randperm(len(cls_patterns))].tolist()
            for cls_patterns in classes_to_patterns_idx
        ]

    # Get PDF values of normal distribution evaluated on range (-4*sigma, 4*sigma)
    norm_sigma = 1
    norm_distribution = Normal(torch.tensor(0), torch.tensor(norm_sigma))
    normal_pdf_range = norm_distribution.log_prob(
        torch.linspace(
            -sigma_range_constant * norm_sigma,
            sigma_range_constant * norm_sigma,
            2 * n_classes + 1,
        ),
    ).exp()
    """ This variable contains pdf values and has shape of 2*n_classes + 1
    We further will apply rolling window to get matrix (n_eperiments, n_classes) 
    where for each experiement number of elements dedicaated to each class 
    looks like a pdf of normal distrubution with mean around one "main" class
    """

    # Apply rolling window. We start with the middle element of array where
    # pdf has a biggest value. Then shift it left for every next class.
    exp_structure = torch.stack(
        [
            normal_pdf_range[n_classes - class_id : -class_id - 1]
            for class_id in range(n_classes)
        ]
    )

    # Renormalize it over experience axis to ensure that total sum do not exceed
    # total number of element in dataset for concrete class.
    #
    # However, to take into account min_class_patterns_in_exp variable, we first
    # deduct min_class_patterns_in_exp * n_classes from total sum of class
    # and then add min_class_patterns_in_exp to each classes in all experiences.
    exp_structure /= exp_structure.sum(dim=1)[:, None]
    exp_structure *= unique_count[:, None] - min_class_patterns_in_exp * n_classes
    exp_structure = torch.floor(exp_structure).int() + min_class_patterns_in_exp

    # Following the exp_structure definition, assign
    # the actual patterns to each experience.
    #
    # For each experience we assign exactly
    # exp_structure[exp_id][class_id] patterns of
    # class "class_id"
    exp_patterns = [[] for _ in range(n_classes)]
    next_idx_per_class = [0 for _ in range(n_classes)]
    for exp_id in range(n_classes):
        for class_id in range(n_classes):
            start_idx = next_idx_per_class[class_id]
            n_patterns = exp_structure[exp_id][class_id]
            end_idx = start_idx + n_patterns
            exp_patterns[exp_id].extend(
                classes_to_patterns_idx[class_id][start_idx:end_idx]
            )
            next_idx_per_class[class_id] = end_idx

    return ni_benchmark(
        train_dataset=seq_train_dataset,
        test_dataset=seq_test_dataset,
        n_experiences=n_classes,
        task_labels=task_labels,
        fixed_exp_assignment=exp_patterns,
        train_transform=train_transform,
        eval_transform=eval_transform,
        reproducibility_data=reproducibility_data,
    )
