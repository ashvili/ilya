import math
import os

import torch

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
    device = kwargs.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    world_size = world_size or meta.get('world_size', 0)

    torch.set_num_threads(max(math.floor(os.cpu_count() / world_size), 1))

    model.eval()
    model.to(device)

    m = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        data = data.to(device, non_blocking=True)
        output = model(data)
        pred = m(output)
        pred_idx = pred.max(1)[1]

    _result = torch.hstack([data.detach().cpu(), pred_idx.unsqueeze(1).detach().cpu(), pred.detach().cpu()])

    if max(1, total_count) and idx % max(1, math.floor(total_count / 10)) == 0:
        print(
            f'Prediction progress: {idx / total_count * 100}%'
        )

    return _result


def predict(model, data: torch.Tensor, nproc=1, *, device: str | None = None, chunk_size: int = 1024):
    """
    Последовательный почанковый инференс с поддержкой GPU/CPU.
    Избегаем multiprocessing, чтобы не копировать модель между процессами (особенно на CUDA).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    chunks = data.tensor_split(max(1, math.ceil(len(data) / max(1, chunk_size))))
    total = len(chunks)
    out = []
    for i, ch in enumerate(chunks):
        out.append(predict_fn(model, ch, i, total, device=device, world_size=nproc))
    print('predict ended')
    return torch.cat(out)
