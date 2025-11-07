import functools
import datetime
import os


def measure_elapsed_time(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        start = datetime.datetime.now()

        fn(*args, **kwargs)

        end = datetime.datetime.now()
        # minutes, seconds = divmod(int((end - start).total_seconds()), 60)
        print(
            f'\tElapsed: {end - start}'
        )

    return wrapped


import random
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
