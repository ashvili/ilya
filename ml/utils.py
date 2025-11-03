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
