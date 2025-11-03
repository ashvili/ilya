import numpy as np
import pandas as pd

from exploration.grid import Grid
from exploration.ray import Ray


class WellSegment(Ray):
    content: str

    def __str__(self):
        with np.printoptions(precision=3, suppress=True):
            return f'WellSegment {self.origin} -> {self.end}'

    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        super().__init__(origin, direction)
        self.content = ''


class Well:
    name: str
    head: np.array
    segments: list[WellSegment]

    @property
    def end(self):
        if not self.segments:
            return self.head
        return self.segments[-1].end

    def __init__(self, name: str, head):
        self.segments = []

        self.name = name

        if isinstance(head, np.ndarray):
            self.head = head
        else:
            self.head = np.array(head)

        assert self.head.shape == (3,)

    def add_segment(self, segment: WellSegment):
        self.segments.append(segment)

    def __str__(self):
        with np.printoptions(precision=3, suppress=True):
            return f'Well {self.head} -> {self.end}'


class Site(Grid):
    wells: list[Well]

    def __init__(self, voxel_size):
        super().__init__(voxel_size)
        self.wells = []
        self.max_bound = np.array([0, 0, 0], dtype=float)
        self.min_bound = np.array([0, 0, 0], dtype=float)

    def add_well(self, well: Well):
        self.wells.append(well)

    def fit_bounds(self, margin_voxels: int = 1):
        """Compute AABB of all segment endpoints and snap to voxel grid with margin."""
        pts = []
        for w in self.wells:
            for s in w.segments:
                p1 = w.head + s.origin
                p2 = w.head + s.end
                pts.append(p1)
                pts.append(p2)
        if not pts:
            return
        v = float(self.voxel_size)
        pts = np.array(pts, dtype=float)
        mn = np.floor(pts.min(axis=0) / v) * v
        mx = np.ceil(pts.max(axis=0) / v) * v
        pad = margin_voxels * v
        self.min_bound = mn - pad
        self.max_bound = mx + pad
        #
        # if not self.max_bound[0] or well.head[0] > self.max_bound[0]:
        #     self.max_bound[0] = well.head[0] + 100
        # if not self.min_bound[0] or well.head[0] < self.min_bound[0]:
        #     self.min_bound[0] = well.head[0] - 100
        #
        # if not self.max_bound[1] or well.head[1] > self.max_bound[1]:
        #     self.max_bound[1] = well.head[1] + 100
        # if not self.min_bound[1] or well.head[1] < self.min_bound[1]:
        #     self.min_bound[1] = well.head[1] - 100
        #
        # if not self.max_bound[2] or well.head[2] > self.max_bound[2]:
        #     self.max_bound[2] = well.head[2] + 100
        # if not self.min_bound[2] or well.head[2] < self.min_bound[2]:
        #     self.min_bound[2] = well.head[2] - 100

    def __str__(self):
        with np.printoptions(precision=3, suppress=True):
            return f'Site {self.min_bound} -> {self.max_bound}'

    def process_with_norm(self, normalize: str | None = None):
        """
        Трассировка всех сегментов через воксельную сетку (Amanatides–Woo).
        Возвращает DataFrame с блоками:
          x,y,z, content_id, _x,_y,_z  (+ x_n,y_n,z_n при normalize != None)

        normalize:
          - 'voxel'  -> (coord - min_bound) / voxel_size
          - 'minmax' -> (coord - min_bound) / (max_bound - min_bound)
          - None     -> без нормализации
        """
        # если границы не заданы, вычислим по данным (min/max прижаты к сетке, с отступом)
        if not np.any(self.max_bound - self.min_bound):
            self.fit_bounds(margin_voxels=1)

        rows = []
        seen = set()

        for well in self.wells:
            for segment in well.segments:
                ray = Ray(segment.origin + well.head, segment.direction)
                voxel_indices = self.find_voxels(ray, 0.0, 1.0)
                if not voxel_indices:
                    continue

                for ix, iy, iz in voxel_indices:
                    key = (int(ix), int(iy), int(iz),
                           int(segment.content) if str(segment.content).isdigit() else segment.content)
                    if key in seen:
                        continue
                    seen.add(key)

                    x = self.min_bound[0] + ix * self.voxel_size
                    y = self.min_bound[1] + iy * self.voxel_size
                    z = self.min_bound[2] + iz * self.voxel_size

                    rows.append({
                        "x": x, "y": y, "z": z,
                        "content_id": int(segment.content) if str(segment.content).isdigit() else segment.content,
                        "_x": self.voxel_size, "_y": self.voxel_size, "_z": self.voxel_size,
                    })

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        # стабильные типы
        df = df.astype({
            "x": "float64", "y": "float64", "z": "float64",
            "_x": "int16", "_y": "int16", "_z": "int16",
        })

        # нормализация (добавляет x_n, y_n, z_n)
        if normalize is not None:
            min_x, min_y, min_z = self.min_bound
            max_x, max_y, max_z = self.max_bound

            if normalize == "voxel":
                denom_x = float(self.voxel_size) if self.voxel_size != 0 else 1.0
                denom_y = float(self.voxel_size) if self.voxel_size != 0 else 1.0
                denom_z = float(self.voxel_size) if self.voxel_size != 0 else 1.0

            elif normalize == "minmax":
                # избегаем деления на 0, если рамка по оси выродилась
                denom_x = float(max_x - min_x) if max_x > min_x else 1.0
                denom_y = float(max_y - min_y) if max_y > min_y else 1.0
                denom_z = float(max_z - min_z) if max_z > min_z else 1.0

            else:
                raise ValueError("normalize must be one of: 'voxel', 'minmax', None")

            df["x_n"] = (df["x"] - min_x) / denom_x
            df["y_n"] = (df["y"] - min_y) / denom_y
            df["z_n"] = (df["z"] - min_z) / denom_z

            # приводим тип к float64 (на случай смешанных типов)
            df[["x_n", "y_n", "z_n"]] = df[["x_n", "y_n", "z_n"]].astype("float64")

        return df

    def process(self):
        # если границы не заданы, вычислим по данным
        if not np.any(self.max_bound - self.min_bound):
            self.fit_bounds(margin_voxels=1)

        rows = []
        seen = set()
        for well in self.wells:
            for segment in well.segments:
                ray = Ray(segment.origin + well.head, segment.direction)
                voxel_indices = self.find_voxels(ray, 0, 1)
                if not voxel_indices:
                    continue
                for ix, iy, iz in voxel_indices:
                    key = (int(ix), int(iy), int(iz), int(segment.content) if str(segment.content).isdigit() else segment.content)
                    # объединяем для теста группы 8..12 в 8 (Other)
                    _raw = int(segment.content) if str(segment.content).isdigit() else segment.content
                    cid = 8 if isinstance(_raw, int) and _raw >= 8 else _raw

                    key = (int(ix), int(iy), int(iz), cid)

                    if key in seen:
                        continue
                    seen.add(key)
                    x = self.min_bound[0] + ix * self.voxel_size
                    y = self.min_bound[1] + iy * self.voxel_size
                    z = self.min_bound[2] + iz * self.voxel_size
                    rows.append({
                        'x': x, 'y': y, 'z': z,
                        # 'content_id': int(segment.content) if str(segment.content).isdigit() else segment.content,
                        'content_id': cid,
                        'well_id': getattr(well, 'name', None),
                        '_x': self.voxel_size, '_y': self.voxel_size, '_z': self.voxel_size
                    })

        df = pd.DataFrame(rows)
        if not df.empty:
            # for c in ['_x','_y','_z']:
            #     df[c] = df[c].astype('int16')
            # if np.issubdtype(df['x'].dtype, np.number):
            #     df[['x','y','z']] = df[['x','y','z']].astype('float64')
            # стабильные типы
            df[['_x', '_y', '_z']] = df[['_x', '_y', '_z']].astype('int16')
            df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype('float64')
            # content_id теперь 1..8

            if pd.api.types.is_integer_dtype(df['content_id']) or pd.api.types.is_float_dtype(df['content_id']):
                df['content_id'] = df['content_id'].astype('int32')
            # well_id как строка (id скважины)

            if 'well_id' in df.columns:
                df['well_id'] = df['well_id'].astype('string')

        return df
