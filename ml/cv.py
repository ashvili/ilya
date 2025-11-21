# ml/cv.py
from __future__ import annotations
import os
import contextlib
from dataclasses import dataclass, asdict
from typing import Iterable, List, Dict, Tuple, Optional, Set
import random

import torch
import pandas as pd

from .dataset import get_datasets
from .model.module import NeuralNetwork
from .model.train import train
from .model.test import test_epoch   # предполагается, что в test.py есть функция test_epoch(model, dataloader/device/...)
from .utils import set_global_seed
from torch.utils.data import DataLoader
import torch.nn as nn
from config import settings

from tqdm.auto import tqdm
_TQDM = tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательный контекстный менеджер, чтобы временно проставлять env-переменные
# и не ломать глобальное окружение проекта.
# ─────────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def patched_env(**kwargs):
    old = {}
    try:
        for k, v in kwargs.items():
            old[k] = os.environ.get(k)
            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Метрики per-well (только по строкам этой скважины в тест-наборе)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WellMetrics:
    well_id: str | int
    n_blocks: int
    n_errors: int
    rel_error_pct: float  # 100 * n_errors / max(n_blocks,1)
    acc: float
    f1_micro: float
    f1_macro: float


def compute_per_well_metrics(
    model: torch.nn.Module,
    test_dataset,
    device: str = "cpu",
) -> List[WellMetrics]:
    """
    Вычисляет метрики отдельно для каждой скважины, присутствующей в ТЕСТ-наборе.
    Предполагает, что test_dataset.__getitem__ -> (X, y_idx) и test_dataset.well_id выровнен по индексам.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    # Сгруппируем индексы теста по well_id
    df_w = pd.DataFrame({"idx": range(len(test_dataset)), "well": test_dataset.well_id})
    groups = df_w.groupby("well", sort=False)["idx"].apply(list)

    results: List[WellMetrics] = []
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for well, idx_list in groups.items():
            if not idx_list:
                continue
            # Соберём мини-DataLoader по индексам этой скважины
            X_list, y_list = [], []
            for i in idx_list:
                X, y_idx = test_dataset[i]
                X_list.append(X.unsqueeze(0))
                y_list.append(int(y_idx))

            Xb = torch.cat(X_list, dim=0).to(device)
            yb = torch.tensor(y_list, dtype=torch.long, device=device)

            logits = model(Xb)
            pred = logits.argmax(dim=1)

            # базовые метрики
            n = yb.numel()
            n_err = int((pred != yb).sum().item())
            acc = float((pred == yb).sum().item() / max(n, 1))

            # f1 micro/macro (локально по скважине)
            num_classes = int(max(yb.max().item(), pred.max().item())) + 1 if n > 0 else 0
            cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
            for t, p in zip(yb, pred):
                cm[t, p] += 1
            tp = cm.diag().to(torch.float32)
            fp = cm.sum(dim=0).to(torch.float32) - tp
            fn = cm.sum(dim=1).to(torch.float32) - tp
            precision = tp / torch.clamp(tp + fp, min=1.0)
            recall    = tp / torch.clamp(tp + fn, min=1.0)
            f1_per_class = 2 * precision * recall / torch.clamp(precision + recall, min=1e-12)
            f1_macro = f1_per_class.mean().item() if num_classes > 0 else 0.0

            tp_micro = tp.sum()
            fp_micro = fp.sum()
            fn_micro = fn.sum()
            precision_micro = float(tp_micro / max(tp_micro + fp_micro, 1.0))
            recall_micro    = float(tp_micro / max(tp_micro + fn_micro, 1.0))
            f1_micro = 0.0
            denom = (precision_micro + recall_micro)
            if denom > 0:
                f1_micro = float(2 * precision_micro * recall_micro / denom)

            results.append(WellMetrics(
                well_id=well,
                n_blocks=int(n),
                n_errors=int(n_err),
                rel_error_pct=100.0 * n_err / max(n, 1),
                acc=float(acc),
                f1_micro=float(f1_micro),
                f1_macro=float(f1_macro),
            ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Основные сценарии: k–1/1 и k–x/x
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FoldSummary:
    holdout_wells: List[str | int]
    test_loss: float
    test_acc: float
    f1_micro: float
    f1_macro: float
    per_well: List[WellMetrics]


def _evaluate_on_test(model, test_dataset, device="cpu", batch_size=2048, num_workers=0):
    # быстрая глобальная оценка на всем тесте (как в твоём test.py),
    # здесь можно использовать готовый test_epoch, если он возвращает нужные метрики
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # ожидается, что test_epoch вернет: avg_loss, accuracy, f1_micro, f1_macro
    # передаем loss_fn явно (у нас таргеты — индексы классов)
    loss_fn = nn.CrossEntropyLoss()
    avg_loss, correct, count, f1_micro, f1_macro = test_epoch(
        model,
        loader,
        device=device,
        loss_fn=loss_fn,
    )
    accuracy = float(correct / max(count, 1))
    return float(avg_loss), accuracy, float(f1_micro), float(f1_macro)


def run_k_minus_1(
    csv_path: str,
    *,
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    device: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    LOXO (k–1/1): на каждом фолде выколотая ровно 1 скважина.
    Возвращает:
      - df_folds: сводка по фолдам (глобальные метрики),
      - df_per_well: объединённый отчёт по каждой скважине (acc, f1, rel_error).
    """
    set_global_seed(seed)
    # соберём уникальные скважины по исходному CSV
    df_all = pd.read_csv(csv_path, usecols=["well_id"])
    uniq_wells = sorted(pd.unique(df_all["well_id"]), key=lambda x: str(x))

    fold_rows = []
    per_well_rows = []

    fold_iter = uniq_wells
    if _TQDM and os.getenv("DISABLE_TQDM","0") != "1":
        fold_iter = _TQDM(uniq_wells, desc="[cv k-1] folds", leave=True, ncols=100, mininterval=0.1)

    for w in fold_iter:
        holdout_list = [w]
        # временно подставляем окружение для k–1/1
        with patched_env(HOLDOUT_WELLS_LIST=",".join(map(str, holdout_list))):
            train_ds, test_ds = get_datasets(csv_path)

        # создаём и учим модель
        n_classes = len(train_ds.contents)
        model = NeuralNetwork(output_count=n_classes)

        train(
            model,
            train_ds,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
        )

        # Сохраняем веса модели для этого фолда, если указан путь
        model_output_path = settings.get("ml.model_output_path")
        if model_output_path:
            root, ext = os.path.splitext(model_output_path)
            ext = ext or ".pt"
            save_path = f"{root}_fold_{w}{ext}"
            try:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            except Exception:
                pass
            torch.save(model.state_dict(), save_path)

        # глобальная оценка
        avg_loss, acc, f1_micro, f1_macro = _evaluate_on_test(model, test_ds, device=device)
        if _TQDM and os.getenv("DISABLE_TQDM","0") != "1":
            fold_iter.set_postfix(loss=float(avg_loss), acc=float(acc), f1=float(f1_macro))

        # per-well метрики (по каждой скважине в этом тесте)
        per_w = compute_per_well_metrics(model, test_ds, device=device)

        fold_rows.append({
            "holdout_wells": holdout_list,
            "test_loss": avg_loss,
            "test_acc": acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
        })
        for wm in per_w:
            row = asdict(wm)
            row["fold_holdout"] = str(w)
            per_well_rows.append(row)

    df_folds = pd.DataFrame(fold_rows)
    df_per_well = pd.DataFrame(per_well_rows)

    # Можно добавить сводную по скважинам (средние метрики скважины по всем разам, когда она была в тесте — в LOXO это 1 раз)
    return df_folds, df_per_well


def run_k_minus_x(
    csv_path: str,
    *,
    x_count: Optional[int] = None,        # например, 5
    x_percent: Optional[float] = None,    # например, 10 (% от k)
    repeats: int = 5,                     # сколько случайных прогонов
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    device: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    k–x/x: на каждом фолде выколотые x скважин (или p%).
    Делаем несколько случайных прогонов (repeats), усредняем.
    Возвращает:
      - df_folds: глобальные метрики по каждому фолду,
      - df_per_well: метрики по скважинам (каждый раз, когда скважина попала в тест).
    """
    set_global_seed(seed)
    df_all = pd.read_csv(csv_path, usecols=["well_id"])
    uniq_wells = sorted(pd.unique(df_all["well_id"]), key=lambda x: str(x))
    k = len(uniq_wells)
    assert k >= 2, "Нужно минимум 2 скважины для k–x/x"

    # определяем сколько скважин в тесте
    if x_count is None and x_percent is None:
        # дефолт: 20% скважин
        x_count = max(1, min(k - 1, int(round(0.20 * k))))
    if x_count is None and x_percent is not None:
        x_count = max(1, min(k - 1, int(round((x_percent / 100.0) * k))))

    rng = random.Random(seed)

    fold_rows = []
    per_well_rows = []

    rep_iter = range(repeats)
    if _TQDM and os.getenv("DISABLE_TQDM","0") != "1":
        rep_iter = _TQDM(range(repeats), desc="[cv k-x] repeats", leave=True, ncols=100, mininterval=0.1)

    for r in rep_iter:
        wells = uniq_wells[:]
        rng.shuffle(wells)
        holdout_list = wells[:x_count]

        with patched_env(HOLDOUT_WELLS_LIST=",".join(map(str, holdout_list))):
            train_ds, test_ds = get_datasets(csv_path)

        n_classes = len(train_ds.contents)
        model = NeuralNetwork(output_count=n_classes)

        train(
            model,
            train_ds,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + r,   # небольшое смещение, если хочешь разные сиды по повторам
            device=device,
        )

        # Сохраняем веса модели для этого повтора, если указан путь
        model_output_path = settings.get("ml.model_output_path")
        if model_output_path:
            root, ext = os.path.splitext(model_output_path)
            ext = ext or ".pt"
            save_path = f"{root}_rep_{r}{ext}"
            try:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            except Exception:
                pass
            torch.save(model.state_dict(), save_path)

        avg_loss, acc, f1_micro, f1_macro = _evaluate_on_test(model, test_ds, device=device)
        if _TQDM and os.getenv("DISABLE_TQDM","0") != "1":
            rep_iter.set_postfix(loss=float(avg_loss), acc=float(acc), f1=float(f1_macro))        
        per_w = compute_per_well_metrics(model, test_ds, device=device)

        fold_rows.append({
            "repeat": r,
            "holdout_wells": holdout_list,
            "test_loss": avg_loss,
            "test_acc": acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
        })
        for wm in per_w:
            row = asdict(wm)
            row["repeat"] = r
            per_well_rows.append(row)

    df_folds = pd.DataFrame(fold_rows)
    df_per_well = pd.DataFrame(per_well_rows)
    return df_folds, df_per_well