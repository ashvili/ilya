import math
import numpy as np


class Ray:
    origin: np.ndarray
    direction: np.ndarray

    @property
    def end(self):
        return self.origin + self.direction

    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin
        self.direction = direction

        assert self.origin.shape == (3,)
        assert self.direction.shape == (3,)

    @classmethod
    def from_spherical(cls, origin: np.ndarray, azimuth: float, zenith: float, radial: float):
        direction = np.array(
            [
                radial * math.sin(math.radians(zenith + 90)) * math.cos(math.radians(azimuth - 90)),
                radial * math.sin(math.radians(zenith + 90)) * math.sin(math.radians(azimuth + 90)),
                - radial * math.cos(math.radians(zenith + 90)),
            ]
        )

        return cls(origin, np.round(direction, 6))

    def __str__(self):
        with np.printoptions(precision=3, suppress=True):
            return f'Ray {self.origin} -> {self.end}'
