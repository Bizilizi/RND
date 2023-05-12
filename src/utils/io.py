import pathlib


def create_folders_if_not_exists(*folders):
    for folder in folders:
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
