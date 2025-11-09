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
    device = kwargs.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    test_loss = 0
    correct = 0
    count = 0

    all_true = []
    all_pred = []

    with torch.no_grad():
        for data, target_idx in data_loader:
            data = data.to(device, non_blocking=True)
            target_idx = target_idx.to(device, non_blocking=True)
            output = model(data)
            test_loss += loss_fn(output, target_idx).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target_idx).sum().item()
            count += len(data)
            all_true.append(target_idx.detach().cpu())
            all_pred.append(pred.detach().cpu())

    if len(all_true) > 0:
        y_true = torch.cat(all_true, dim=0)
        y_pred = torch.cat(all_pred, dim=0)
        f1_micro = f1_score(y_true.numpy(), y_pred.numpy(), average='micro')
        f1_macro = f1_score(y_true.numpy(), y_pred.numpy(), average='macro')
    else:
        f1_micro = 0.0
        f1_macro = 0.0

    avg_loss = test_loss / max(count, 1)

    return avg_loss, correct, count, f1_micro, f1_macro
