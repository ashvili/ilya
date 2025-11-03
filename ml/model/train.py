import math
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler

from ml.processes import Event, CallEventParameters, prepare_subprocesses


def train_event(model, data_loader, optimizer, loss_fn, epoch, log_interval):
    return Event(
        name='CALL',
        parameters=CallEventParameters(
            callable=train_epoch,
            kwargs={
                'model': model,
                'data_loader': data_loader,
                'optimizer': optimizer,
                'loss_fn': loss_fn,
                'epoch': epoch,
                'log_interval': log_interval,
            }
        )
    )


def train_epoch(model: nn.Module, data_loader, optimizer, loss_fn,
                epoch, log_interval=100, *args, **kwargs):
    meta = kwargs.get('meta', {'rank': 0, 'world_size': 1})
    rank = meta.get('rank', 0)
    world_size = meta.get('world_size', 0)

    torch.set_num_threads(max(math.floor(os.cpu_count() / world_size), 1))

    model.train()
    pid = os.getpid()

    data_loader.sampler.set_epoch(epoch)
    data_loader.sampler.rank = rank

    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()

        output = model(data.to('cpu'))
        loss = loss_fn(output, target)
        loss.backward()

        optimizer.step()
        if batch_idx % math.floor(log_interval / world_size) == 0:
            if rank in [0, -1]:
                print(
                    f'{pid}\tTrain Epoch: {epoch} '
                    f'[{batch_idx * len(data) * world_size}/{len(data_loader.dataset)} '
                    f'({100. * batch_idx / len(data_loader):.0f}%)]\t\tLoss: {loss.item():.6f}'
                )


def train(model, train_dataset, epochs, batch_size, optimizer_kwargs, nproc=1,
          log_interval=100, output_folder='./temp/'):

    optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)
    loss_class = nn.CrossEntropyLoss
    train_data_loader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        sampler=DistributedSampler(
            dataset=train_dataset,
            num_replicas=nproc,
            rank=0,
        ),
    )

    model.share_memory()

    processes, task_queue, result_queue = prepare_subprocesses(nproc=nproc)

    for epoch in range(1, epochs + 1):
        train_events = []

        for rank in range(nproc):
            train_events.append(train_event(model, train_data_loader, optimizer, loss_class(), epoch, log_interval))

        list(map(task_queue.put, train_events))
        task_queue.join()
        print('train ended')

    for rank in range(nproc):
        task_queue.put(Event(name='STOP'))

    torch.save(model.state_dict(), f'{output_folder}model.pth')
