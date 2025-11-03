import multiprocessing as mp
import os
import typing
from typing import TypedDict


class _BaseCallEventParameters(TypedDict):
    callable: typing.Callable


class CallEventParameters(_BaseCallEventParameters, total=False):
    kwargs: dict


class _BaseEvent(TypedDict):
    name: str


class Event(_BaseEvent, total=False):
    parameters: CallEventParameters


def subprocess_fn(rank: int, world_size: int, task_queue: 'mp.JoinableQueue[Event]', result_queue: mp.Queue = None):
    meta = {
        'rank': rank,
        'world_size': world_size
    }
    print(f'init proc {os.getpid()}')

    while True:
        task = task_queue.get()

        if not task:
            continue

        match task['name']:
            case 'STOP':
                task_queue.task_done()
                break

            case 'CALL':
                result = task['parameters']['callable'](**task['parameters']['kwargs'], meta=meta)
                if result is not None:
                    result_queue.put(result)

            case _ as unsupported_name:
                raise ValueError(f'unsupported event name {unsupported_name}')

        task_queue.task_done()

    print(f'stop proc {os.getpid()}')


def prepare_subprocesses(nproc):
    mp.set_start_method('spawn', force=True)

    task_queue: 'mp.JoinableQueue[Event]' = mp.JoinableQueue()
    result_queue = mp.Queue()
    processes = []

    for rank in range(nproc):
        p = mp.Process(
            target=subprocess_fn,
            kwargs={
                'rank': rank,
                'world_size': nproc,
                'task_queue': task_queue,
                'result_queue': result_queue,
            }
        )
        p.start()
        processes.append(p)

    return processes, task_queue, result_queue
