import os

import pandas as pd
import torch

from ml.model import NeuralNetwork, train, predict
from ml.utils import measure_elapsed_time, set_global_seed
from ml.dataset import get_datasets
from config import settings

from .cv import run_k_minus_1, run_k_minus_x

# ─────────────────────────────────────────────────────────────
# Диагностика разбиения и нормализации (очень полезно перед обучением)
# ─────────────────────────────────────────────────────────────
def _summarize_split(train_ds, test_ds):
    # 1) Список скважин в train/test
    import numpy as np
    train_wells = list(pd.unique(pd.Series(train_ds.well_id)))
    test_wells  = list(pd.unique(pd.Series(test_ds.well_id)))
    train_wells_sorted = sorted(set(train_wells), key=lambda x: str(x))
    test_wells_sorted  = sorted(set(test_wells),  key=lambda x: str(x))
    union_wells = sorted(set(train_wells_sorted) | set(test_wells_sorted), key=lambda x: str(x))

    print("\n[split] ─────────────────────────────────────")
    print(f"[split] total wells: {len(union_wells)}")
    print(f"[split] train wells (unique): {len(train_wells_sorted)}")
    print(f"[split] test  wells (unique): {len(test_wells_sorted)}")
    print(f"[split] train wells: {train_wells_sorted}")
    print(f"[split] test  wells: {test_wells_sorted}")

    # 2) Границы нормализации (MIN/MAX) — ОБЯЗАТЕЛЬНО из TRAIN
    print("\n[norm] MIN/MAX (computed on TRAIN, applied to both):")
    try:
        borders = getattr(train_ds, "borders", None) or {}
        for axis in ("X_x", "X_y", "X_z"):
            b = borders.get(axis, {})
            vmin = b.get("MIN", None)
            vmax = b.get("MAX", None)
            if vmin is None or vmax is None:
                print(f"[norm] {axis}: <no borders>")
            else:
                # печатаем с 6 знаками — достаточно для контроля
                print(f"[norm] {axis}: min={float(vmin):.6f}, max={float(vmax):.6f}")
    except Exception as e:
        print(f"[norm] failed to print borders: {e}")

    # 3) Список классов (порядок критичен — он задаёт индексацию)
    print("\n[classes] mapping (idx → label) used for training/prediction:")
    try:
        contents = getattr(train_ds, "contents", None) or {}
        class_names = [contents[i] for i in range(len(contents))]
        print(f"[classes] count={len(class_names)}")
        print(f"[classes] {class_names}")
    except Exception as e:
        print(f"[classes] failed to print class mapping: {e}")
    print("────────────────────────────────────────────\n")

@measure_elapsed_time
def main(datasets_filepath,
         epochs=10,
         batch_size=64,
         lr=1e-3,
         nproc=1,
         log_interval=50,
         output_folder='./temp/'):

    # ─────────────────────────────────────────────────────────────
    # ВАЖНО: фиксируем сиды ДО чтения данных и выбора holdout-скважин,
    # чтобы сплиты/инициализации были воспроизводимыми.
    # ─────────────────────────────────────────────────────────────
    set_global_seed(settings.get("ml.holdout.seed", 42))

    # 2️⃣ — Обычный режим (без кросс-валидации)


    print(f'{nproc=}')

    train_dataset, test_dataset = get_datasets(datasets_filepath)
    _summarize_split(train_dataset, test_dataset)
    n_classes = len(train_dataset.contents)
    model = NeuralNetwork(n_classes).to('cpu')

    train(
        model,
        train_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        seed=settings.get("ml.holdout.seed", 42),
    )
    print('train finished')

    # Save trained weights if configured
    model_output_path = settings.get("ml.model_output_path")
    if model_output_path:
        try:
            os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
        except Exception:
            pass
        torch.save(model.state_dict(), model_output_path)
        print(f"Saved model weights to {os.path.abspath(model_output_path)}")

    x_min, x_max = train_dataset.borders['X_x']['MIN'], train_dataset.borders['X_x']['MAX']
    y_min, y_max = train_dataset.borders['X_y']['MIN'], train_dataset.borders['X_y']['MAX']
    z_min, z_max = train_dataset.borders['X_z']['MIN'], train_dataset.borders['X_z']['MAX']

    block_size = 5

    # normalize coords
    _tx = torch.arange(x_min, x_max, step=block_size)
    _ty = torch.arange(y_min, y_max, step=block_size)
    _tz = torch.arange(z_min, z_max, step=block_size)

    n_tx = torch.div(torch.sub(_tx, min(_tx)), (max(_tx) - min(_tx)))
    n_ty = torch.div(torch.sub(_ty, min(_ty)), (max(_ty) - min(_ty)))
    n_tz = torch.div(torch.sub(_tz, min(_tz)), (max(_tz) - min(_tz)))

    # n_tx = n_tx[:100]
    # n_ty = n_ty[:100]
    # n_tz = n_tz[:100]

    print('Cartesian product started')
    predict_data = torch.cartesian_prod(n_tx, n_ty, n_tz)
    print('Cartesian product finished')

    # predict
    print('Prediction started')
    result = predict(model, predict_data, nproc)
    print('Prediction finished')

    # denormalize
    result[:, 0] = torch.add(torch.mul(result[:, 0], (x_max - x_min)), x_min)
    result[:, 1] = torch.add(torch.mul(result[:, 1], (y_max - y_min)), y_min)
    result[:, 2] = torch.add(torch.mul(result[:, 2], (z_max - z_min)), z_min)

    df = pd.DataFrame(result.numpy())

    df.iloc[:, 3] = df.iloc[:, 3].astype(int).map(train_dataset.contents)

    class_names = [train_dataset.contents[i] for i in range(len(train_dataset.contents))]
    extra_columns = [f'prob_{name}' for name in class_names]

    df.columns = ['x', 'y', 'z', 'content_id', *extra_columns]
    df['_x'] = block_size
    df['_y'] = block_size
    df['_z'] = block_size

    print('Saving...')
    contents = [d for _, d in df.groupby(['content_id'])]
    ores_count = len(contents)
    for idx, predicted in enumerate(contents):
        c_id = predicted['content_id'].iloc[0]
        fpath = f'{output_folder}predicted_{c_id}.csv'

        predicted.to_csv(fpath, index=False)
        print(f'[ {idx + 1} / {ores_count} ] Done, saved to {os.path.abspath(fpath)}')


if __name__ == '__main__':

    ml_cfg = settings.section("ml")

    # Threads/env tuning from config
    threads = ml_cfg.get("threads", {}) or {}
    if isinstance(threads, dict):
        for k, v in threads.items():
            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[str(k)] = str(v)

    cv_mode = str(ml_cfg.get("cv", "none"))
    blocks_file_path = ml_cfg.get("blocks_file_path")
    epochs = int(ml_cfg.get("epochs", 100))
    batch_size = int(ml_cfg.get("batch_size", 63))
    learning_rate = float(ml_cfg.get("learning_rate", 1e-3))
    output_folder = ml_cfg.get("output_folder", "./temp/")

    if cv_mode == "k-1":
        df_folds, df_per_well = run_k_minus_1(
            blocks_file_path,
            seed=settings.get("ml.holdout.seed", 42),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=None,
        )
        df_folds.to_csv("cv_k-1_folds.csv", index=False)
        df_per_well.to_csv("cv_k-1_per_well.csv", index=False)
        print("Saved cv_k-1_folds.csv and cv_k-1_per_well.csv")
    elif cv_mode == "k-x":
        df_folds, df_per_well = run_k_minus_x(
            blocks_file_path,
            x_count=ml_cfg.get("fold_x"),
            x_percent=ml_cfg.get("fold_x_percent"),
            repeats=int(ml_cfg.get("cv_repeats", 5)),
            seed=settings.get("ml.holdout.seed", 42),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=None,
        )
        df_folds.to_csv("cv_k-x_folds.csv", index=False)
        df_per_well.to_csv("cv_k-x_per_well.csv", index=False)
        print("Saved cv_k-x_folds.csv and cv_k-x_per_well.csv")
    else:
        main(
            blocks_file_path,
            nproc=int(ml_cfg.get("nproc", os.cpu_count() or 1)),
            epochs=epochs,
            batch_size=batch_size,
            lr=learning_rate,
            output_folder=output_folder,
        )
