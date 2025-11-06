import math
import os

import torch
from sklearn.metrics import f1_score
from ml.processes import Event, CallEventParameters


def test_event(model, data_loader, loss_fn):
    return Event(
        name='CALL',
        parameters=CallEventParameters(
            callable=test_epoch,
            kwargs={
                'model': model,
                'data_loader': data_loader,
                'loss_fn': loss_fn,
            }
        )
    )


def test_epoch(model, data_loader, loss_fn, *args, **kwargs):
    meta = kwargs.get('meta', {'rank': 0, 'world_size': 1})
    rank = meta.get('rank', 0)
    world_size = meta.get('world_size', 0)

    torch.set_num_threads(max(math.floor(os.cpu_count() / world_size), 1))

    model.eval()

    data_loader.sampler.rank = rank

    test_loss = 0
    correct = 0
    count = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to('cpu'))
            test_loss += loss_fn(output, target).item()
            pred = output.max(1)[1]
            correct += pred.eq(target.max(1)[1]).sum().item()
            count += len(data)
            f1_micro = f1_score(target.max(1)[1], pred, average='micro')
            f1_macro = f1_score(target.max(1)[1], pred, average='macro')

    avg_loss = test_loss / count

    return avg_loss, correct, count, f1_micro, f1_macro
