# ml/train.py
import torch
from torch.utils.data import DataLoader

def train(
    model,
    dataset,
    *,
    epochs: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    device: str | None = None,
    num_workers: int = 0,
    seed: int = 42,
    **kwargs,  # принимает лишние параметры (nproc и пр.), чтобы не ломать существующие вызовы
):
    """
    Однопроцессное обучение:
      - dataset: возвращает (X, y_idx), где y_idx — индексы классов (LongTensor)
      - CrossEntropyLoss по логитам
      - детерминированный DataLoader (generator с seed)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # фиксируем генератор для воспроизводимого shuffle
    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0

        for X, y_idx in loader:
            X = X.to(device)
            y_idx = y_idx.to(device)  # индексы классов (B,)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X)         # (B, C)
            loss = loss_fn(logits, y_idx)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            count += X.size(0)

        avg_loss = running_loss / max(count, 1)
        print(f"[train] epoch {epoch}/{epochs}  loss={avg_loss:.6f}")

    return model
