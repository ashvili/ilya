import math
import os

import torch

import multiprocessing
from ml.processes import Event, CallEventParameters, prepare_subprocesses


def predict_event(model, data, idx, total_count):
    return Event(
        name='CALL',
        parameters=CallEventParameters(
            callable=predict_fn,
            kwargs={
                'model': model,
                'data': data,
                'idx': idx,
                'total_count': total_count,
            }
        )
    )


def predict_fn(model, data, idx, total_count, world_size=0, *args, **kwargs):
    meta = kwargs.get('meta', {'rank': 0, 'world_size': 1})

    world_size = world_size or meta.get('world_size', 0)

    torch.set_num_threads(max(math.floor(os.cpu_count() / world_size), 1))

    model.eval()

    m = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        output = model(data.to('cpu'))
        pred = m(output)
        pred_idx = pred.max(1)[1]

    _result = torch.hstack([data, pred_idx.unsqueeze(1), pred])

    if idx % math.floor(100) == 0:
        print(
            f'Prediction progress: {idx / total_count * 100}%'
        )

    return _result


def predict(model, data: torch.Tensor, nproc=1):
    import multiprocessing

    splitted_data = data.tensor_split(math.ceil(len(data) / 1024))
    total_count = len(splitted_data)

    with multiprocessing.Pool(nproc) as p:
        predicted = p.starmap(predict_fn, [
            (model, splitted_datum, idx, total_count, nproc)
            for idx, splitted_datum
            in enumerate(splitted_data)
        ])
        p.close()
        p.join()

    print('predict ended')

    tensor = torch.cat(predicted)

    return tensor
