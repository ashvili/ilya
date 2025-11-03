import math
import warnings

import numpy as np

from exploration import ray
from exploration.ray import Ray


class Grid:
    min_bound: np.ndarray
    max_bound: np.ndarray
    voxel_size: int

    def __init__(self, voxel_size: int):
        self.min_bound = np.array([0, 0, 0], dtype=float)
        self.max_bound = np.array([0, 0, 0], dtype=float)

        self.voxel_size = int(voxel_size)

    def ray_grid_intersection(self, ray: Ray):

        def get_inv_direction(_direction):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.float64(1) / np.float64(_direction)

        x_inv_dir = get_inv_direction(ray.direction[0])

        if x_inv_dir >= 0:
            t_min = (self.min_bound[0] - ray.origin[0]) * x_inv_dir
            t_max = (self.max_bound[0] - ray.origin[0]) * x_inv_dir
        else:
            t_min = (self.max_bound[0] - ray.origin[0]) * x_inv_dir
            t_max = (self.min_bound[0] - ray.origin[0]) * x_inv_dir

        y_inv_dir = get_inv_direction(ray.direction[1])

        if y_inv_dir >= 0:
            t_y_min = (self.min_bound[1] - ray.origin[1]) * y_inv_dir
            t_y_max = (self.max_bound[1] - ray.origin[1]) * y_inv_dir
        else:
            t_y_min = (self.max_bound[1] - ray.origin[1]) * y_inv_dir
            t_y_max = (self.min_bound[1] - ray.origin[1]) * y_inv_dir

        if t_min > t_y_max or t_y_min > t_max:
            return False

        if t_y_min > t_min:
            t_min = t_y_min
        if t_y_max < t_max:
            t_max = t_y_max

        z_inv_dir = get_inv_direction(ray.direction[2])

        if z_inv_dir >= 0:
            t_z_min = (self.min_bound[2] - ray.origin[2]) * z_inv_dir
            t_z_max = (self.max_bound[2] - ray.origin[2]) * z_inv_dir
        else:
            t_z_min = (self.max_bound[2] - ray.origin[2]) * z_inv_dir
            t_z_max = (self.min_bound[2] - ray.origin[2]) * z_inv_dir

        if t_min > t_z_max or t_z_min > t_max:
            return False

        if t_z_min > t_min:
            t_min = t_z_min
        if t_z_max < t_max:
            t_max = t_z_max

        return t_min, t_max
        # return max(t_min, 0), max(t_max, 0)

    """
    Amanatides & Woo voxel traversal with zero-based indices and no off-by-one.
    Start cell = floor((p - min_bound)/voxel_size) for each axis.
    """
    def amanatides_woo_algorithm(self, ray: Ray, t_min, t_max):
        ray_start = ray.origin + ray.direction * t_min
        ray_end = ray.origin + ray.direction * t_max

        # zero-based voxel indices
        ix = math.floor((ray_start[0] - self.min_bound[0]) / self.voxel_size)
        iy = math.floor((ray_start[1] - self.min_bound[1]) / self.voxel_size)
        iz = math.floor((ray_start[2] - self.min_bound[2]) / self.voxel_size)

        ex = math.floor((ray_end[0] - self.min_bound[0]) / self.voxel_size)
        ey = math.floor((ray_end[1] - self.min_bound[1]) / self.voxel_size)
        ez = math.floor((ray_end[2] - self.min_bound[2]) / self.voxel_size)

        # step and tMax / tDelta per axis
        if ray.direction[0] > 0:
            step_x = 1
            next_boundary_x = self.min_bound[0] + (ix + 1) * self.voxel_size
            t_max_x = t_min + (next_boundary_x - ray_start[0]) / ray.direction[0]
            t_dx = self.voxel_size / ray.direction[0]
        elif ray.direction[0] < 0:
            step_x = -1
            next_boundary_x = self.min_bound[0] + ix * self.voxel_size
            t_max_x = t_min + (next_boundary_x - ray_start[0]) / ray.direction[0]
            t_dx = -self.voxel_size / ray.direction[0]  # positive
        else:
            step_x = 0
            t_max_x = float('inf')
            t_dx = float('inf')

        if ray.direction[1] > 0:
            step_y = 1
            next_boundary_y = self.min_bound[1] + (iy + 1) * self.voxel_size
            t_max_y = t_min + (next_boundary_y - ray_start[1]) / ray.direction[1]
            t_dy = self.voxel_size / ray.direction[1]
        elif ray.direction[1] < 0:
            step_y = -1
            next_boundary_y = self.min_bound[1] + iy * self.voxel_size
            t_max_y = t_min + (next_boundary_y - ray_start[1]) / ray.direction[1]
            t_dy = -self.voxel_size / ray.direction[1]
        else:
            step_y = 0
            t_max_y = float('inf')
            t_dy = float('inf')

        if ray.direction[2] > 0:
            step_z = 1
            next_boundary_z = self.min_bound[2] + (iz + 1) * self.voxel_size
            t_max_z = t_min + (next_boundary_z - ray_start[2]) / ray.direction[2]
            t_dz = self.voxel_size / ray.direction[2]
        elif ray.direction[2] < 0:
            step_z = -1
            next_boundary_z = self.min_bound[2] + iz * self.voxel_size
            t_max_z = t_min + (next_boundary_z - ray_start[2]) / ray.direction[2]
            t_dz = -self.voxel_size / ray.direction[2]
        else:
            step_z = 0
            t_max_z = float('inf')
            t_dz = float('inf')

        blocks = [(ix, iy, iz)]

        # traverse until we reach the end voxel
        while not (ix == ex and iy == ey and iz == ez):
            if t_max_x <= t_max_y and t_max_x <= t_max_z:
                ix += step_x
                t_max_x += t_dx
            elif t_max_y <= t_max_z:
                iy += step_y
                t_max_y += t_dy
            else:
                iz += step_z
                t_max_z += t_dz
            blocks.append((ix, iy, iz))

        return blocks

    def find_voxels(self, ray: Ray, t_min, t_max):
        intersections = self.ray_grid_intersection(ray)

        if not intersections:
            return

        t_ray_min, t_ray_max = intersections
        t_min = max(t_min, t_ray_min)
        t_max = min(t_max, t_ray_max)

        if t_min > t_max:
            return

        voxel_indices = self.amanatides_woo_algorithm(ray, t_min, t_max)

        return voxel_indices
