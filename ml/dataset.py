import math
import os
import random
from typing import Optional, Set, Dict, List

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class Lithology(Dataset):
    """
    Датасет с поддержкой:
      • чтения well_id,
      • разбиения k–x/x по скважинам,
      • НОРМАЛИЗАЦИИ ПО min/max ИМЕННО ТРЕЙНА (границы приходят извне),
      • единого словаря классов для согласованности индексов train/test.
    """
    train = True

    data: torch.Tensor = None
    target: torch.Tensor = None
    well_id = None

    borders: Dict[str, Dict[str, float]] = None  # {'X_x': {'MIN':..., 'MAX':...}, ...}
    data_column_names = ['X_x', 'X_y', 'X_z']
    target_column_name = 'Y'
    contents: Dict[int, str] = None  # idx -> original label

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        train: bool = True,
        holdout_wells: Optional[Set] = None,
        borders: Optional[Dict[str, Dict[str, float]]] = None,
        classes: Optional[List[str]] = None
    ):
        super().__init__()
        self.train = train
        self.holdout_wells = holdout_wells or set()
        self.borders = borders  # если None — (теоретически) могли бы посчитать тут, но мы даём извне
        self._prepare_tensors(df, classes)

    def __getitem__(self, index) -> tuple:
        return self.data[index], self.target[index]

    def __len__(self):
        return self.data.shape[0]

    # --- helpers ---

    def _undublicate(self, df: pd.DataFrame) -> pd.DataFrame:
        # удаляем дубликаты по координатам признаков
        return df.drop_duplicates(subset=self.data_column_names, ignore_index=True)

    def _normalize_data(self, df: pd.DataFrame) -> torch.Tensor:
        # нормализация по заранее переданным границам train-сета
        for _cn in self.data_column_names:
            _min = self.borders[_cn]['MIN']
            _max = self.borders[_cn]['MAX']
            denom = (_max - _min)
            if denom == 0:  # защита от деления на ноль, если все значения по оси равны
                df[_cn] = 0.0
            else:
                df[_cn] = (df[_cn] - _min) / denom
        tensor = torch.tensor(df[self.data_column_names].values)
        return tensor.type(torch.FloatTensor)

    def _normalize_target(self, df: pd.DataFrame, classes: Optional[List[str]]) -> torch.Tensor:
        # единый словарь классов (train/test согласованы)
        if classes is not None:
            class_to_idx = {c: i for i, c in enumerate(classes)}
            idx_series = df[self.target_column_name].map(class_to_idx)
            if idx_series.isna().any():
                # если вдруг во фрейме появились классы, которых нет в предоставленном списке
                unknown = df[self.target_column_name][idx_series.isna()].unique()
                raise ValueError(f"Unknown classes in dataset not present in provided class list: {unknown}")
            factorized = idx_series.astype(int).to_numpy()
            index = classes
        else:
            # fallback (не рекомендуется): факторизуем локально
            factorized, index = pd.factorize(df[self.target_column_name])
        self.contents = {i: content for i, content in enumerate(index)}
        tfactorized = torch.tensor(factorized, dtype=torch.long)

                # возвращаем индексы классов (LongTensor)
        self.class_count = len(self.contents)
        return tfactorized

    def _prepare_tensors(self, df_raw: pd.DataFrame, classes: Optional[List[str]]):
        # Переименуем колонки в рабочие имена
        df = df_raw.rename(columns={
            'x': 'X_x',
            'y': 'X_y',
            'z': 'X_z',
            'content_id': 'Y',
        }).copy()

        # Удалим дубликаты по координатам (до маски — чтобы индексы well_id соответствовали)
        df = self._undublicate(df)

        # Сохраняем well_id (до маскирования, чтобы маски применялись синхронно)
        well_ids_series = df['well_id'].reset_index(drop=True)

        # Формируем маски train/test по выбранным скважинам (k–x/x)
        if self.holdout_wells:
            mask_test = well_ids_series.isin(self.holdout_wells)
            mask_train = ~mask_test
        else:
            # теоретический fallback, но в практике get_datasets всегда передаёт holdout
            cutoff = math.floor(0.80 * len(df))
            mask_train = pd.Series([True] * cutoff + [False] * (len(df) - cutoff))
            mask_test = ~mask_train

        # Делаем подмножество по роли
        role_mask = mask_train if self.train else mask_test
        df_role = df[role_mask].reset_index(drop=True)
        well_role = well_ids_series[role_mask].to_numpy()

        # Нормализация по train-минимакс: границы должны прийти извне
        if self.borders is None:
            raise ValueError("borders (min/max from train) must be provided to Lithology")

        data = self._normalize_data(df_role)
        target = self._normalize_target(df_role, classes)

        self.data = data
        self.target = target
        self.well_id = well_role


def _choose_holdout_wells(df: pd.DataFrame, *, wells_col: str = 'well_id') -> Optional[Set]:
    """
    Выбор множества скважин для holdout (тест) по переменным окружения:
      - HOLDOUT_WELLS_LIST: "W1,W7,W42"
      - HOLDOUT_WELLS_COUNT: "1", "5", ...
      - HOLDOUT_WELLS_PERCENT: "5", "10", ... (0..100)
      - HOLDOUT_SEED: "42" (по умолчанию 42)
    Приоритет: LIST > COUNT > PERCENT. Если ничего не задано — ~20% скважин в тест.
    """
    uniq = list(pd.unique(df[wells_col]))
    k = len(uniq)
    if k == 0:
        return None

    env_list = os.getenv('HOLDOUT_WELLS_LIST')
    if env_list:
        cand = [w.strip() for w in env_list.split(',') if w.strip()]
        return set(cand)

    seed = int(os.getenv('HOLDOUT_SEED', '42'))
    rng = random.Random(seed)
    rng.shuffle(uniq)

    env_count = os.getenv('HOLDOUT_WELLS_COUNT')
    if env_count:
        try:
            cnt = max(1, min(k - 1, int(env_count)))
        except Exception:
            cnt = 1
        return set(uniq[:cnt])

    env_pct = os.getenv('HOLDOUT_WELLS_PERCENT')
    if env_pct:
        try:
            p = float(env_pct)
        except Exception:
            p = 20.0
        p = max(0.0, min(100.0, p))
        cnt = max(1, min(k - 1, round(k * (p / 100.0))))
        return set(uniq[:cnt])

    cnt = max(1, min(k - 1, math.floor(0.20 * k)))
    return set(uniq[:cnt])


def _get_dataframe(_fp: str) -> pd.DataFrame:
    df = pd.read_csv(_fp)
    # читаем обязательные колонки, включая well_id
    df = df[['x', 'y', 'z', 'content_id', 'well_id']]
    # перемешивание (не влияет на разделение по скважинам)
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def _compute_borders_from_train(df: pd.DataFrame, holdout: Set) -> Dict[str, Dict[str, float]]:
    """
    Считаем MIN/MAX ТОЛЬКО по train-скважинам.
    df здесь ещё в исходных колонках; переименуем временно.
    """
    tmp = df.rename(columns={'x': 'X_x', 'y': 'X_y', 'z': 'X_z'}).copy()
    mask_train = ~tmp['well_id'].isin(holdout) if holdout else pd.Series([True] * len(tmp))
    sub = tmp[mask_train]

    # если вдруг после фильтрации пусто — страхуемся на весь df
    if len(sub) == 0:
        sub = tmp

    borders = {}
    for _cn in ['X_x', 'X_y', 'X_z']:
        borders[_cn] = {'MIN': float(sub[_cn].min()), 'MAX': float(sub[_cn].max())}
    return borders


def _compute_classes(df: pd.DataFrame) -> List[str]:
    """
    Единый список классов для согласованных индексов в train/test.
    Порядок соответствует pd.factorize по всему df.
    """
    _, index = pd.factorize(df['content_id'])
    # приводим к python-типам, чтобы гарантировать сериализуемость
    return [str(x) if not isinstance(x, (str, int)) else x for x in index.tolist()]


def get_datasets(filepath: str):
    df = _get_dataframe(filepath)

    # 1) Определяем holdout-скважины (x из k)
    holdout = _choose_holdout_wells(df)

    # 2) Считаем границы нормализации ТОЛЬКО по train-скважинам
    borders = _compute_borders_from_train(df, holdout or set())

    # 3) Фиксируем единый список классов по всему df
    classes = _compute_classes(df)

    # 4) Создаем два датасета, которым передаём ОДНИ И ТЕ ЖЕ borders и classes
    train_ds = Lithology(df, train=True, holdout_wells=holdout, borders=borders, classes=classes)
    test_ds = Lithology(df, train=False, holdout_wells=holdout, borders=borders, classes=classes)
    return train_ds, test_ds
