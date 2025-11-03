import math

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class Lithology(Dataset):
    train = True

    data: torch.Tensor = None
    target: torch.Tensor = None

    borders: dict[str, dict[str, float]] = None
    data_column_names = [
        'X_x',
        'X_y',
        'X_z',
    ]
    target_column_name = 'Y'
    contents: dict[int, str] = None

    def _undublicate(self, df):
        return df.drop_duplicates(subset=self.data_column_names, ignore_index=True)

    def _normalize_data(self, df: pd.DataFrame):

        for _cn in self.data_column_names:
            _min = self.borders[_cn]['MIN']
            _max = self.borders[_cn]['MAX']
            df[_cn] = (df[_cn] - _min) / (_max - _min)

        tensor = torch.tensor(df[self.data_column_names].values)

        return tensor.type(torch.FloatTensor)

    def _normalize_target(self, df):
        factorized, index = pd.factorize(df[self.target_column_name])
        self.contents = {
            i: content
            for i, content in enumerate(index)
        }
        tfactorized = torch.tensor(factorized)
        tensor = torch.zeros(tfactorized.size(0), max(factorized) + 1).scatter_(1, tfactorized.unsqueeze(1), 1.)

        return tensor.type(torch.FloatTensor)

    def _prepare_tensors(self, df: pd.DataFrame):
        df = df.rename(
            columns={
                df.columns[0]: 'X_x',
                df.columns[1]: 'X_y',
                df.columns[2]: 'X_z',
                df.columns[3]: 'Y',
            }
        )
        self.borders = {
            _cn: {
                'MIN': df[_cn].min(),
                'MAX': df[_cn].max()
            }
            for _cn
            in self.data_column_names
        }

        df = self._undublicate(df)
        data = self._normalize_data(df)
        target = self._normalize_target(df)

        cutoff = math.floor(0.80 * data.shape[0])

        if self.train:
            self.data = data[:cutoff]
            self.target = target[:cutoff]
        else:
            self.data = data[cutoff:]
            self.target = target[cutoff:]

    def __init__(self, df: pd.DataFrame, train: bool = True):
        self.train = train
        self._prepare_tensors(df)

    def __getitem__(self, index) -> tuple:
        data = self.data[index]
        target = self.target[index]

        return data, target

    def __len__(self):
        return self.data.shape[0]


def _get_dataframe(_fp):
    df = pd.read_csv(_fp)
    df = df[['x', 'y', 'z', 'content_id']]
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def get_datasets(filepath):
    df = _get_dataframe(filepath)

    return Lithology(df), Lithology(df, train=False)
