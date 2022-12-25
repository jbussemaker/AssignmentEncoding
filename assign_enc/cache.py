import os
from appdirs import user_cache_dir


def get_cache_path(sub_path: str = None) -> str:
    cache_folder = user_cache_dir('AssignmentEncoder')
    os.makedirs(cache_folder, exist_ok=True)
    return os.path.join(cache_folder, sub_path) if sub_path is not None else cache_folder
