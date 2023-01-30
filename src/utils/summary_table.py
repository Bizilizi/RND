from collections import defaultdict

import wandb


def log_summary_table_to_wandb(train_stream, test_stream):
    train_table_data = [[], []]
    test_table_data = [[], []]
    rows = ["classes", "num_elements"]

    for i, exp in enumerate(train_stream):
        classes = {el[1] for el in exp.dataset}
        num_elements = len(exp.dataset)

        train_table_data[0].append(",".join(map(str, classes)))
        train_table_data[1].append(str(num_elements))

    for i, exp in enumerate(test_stream):
        classes = {el[1] for el in exp.dataset}
        num_elements = len(exp.dataset)

        test_table_data[0].append(",".join(map(str, classes)))
        test_table_data[1].append(str(num_elements))

    # create a wandb.Table() with corresponding columns
    columns = list(map(lambda x: f"exp_{x}", range(len(train_stream))))
    train_table = wandb.Table(data=train_table_data, columns=columns, rows=rows)
    test_table = wandb.Table(data=test_table_data, columns=columns, rows=rows)

    wandb.log({"summary/train": train_table})
    wandb.log({"summary/test": test_table})
