import json
import copy
from math import floor, ceil
import numpy as np

from corgie.helpers import crop


def get_bcube_from_coords(start_coord, end_coord, coord_mip, cant_be_empty=True):
    xs, ys, zs = [int(i) for i in start_coord.split(",")]
    xe, ye, ze = [int(i) for i in end_coord.split(",")]
    bcube = BoundingCube(xs, xe, ys, ye, zs, ze, coord_mip)

    if cant_be_empty and bcube.area() * bcube.z_size() == 0:
        raise Exception(
            "Attempted creation of an empty bounding "
            "when 'cant_be_empty' flag is set to True"
        )

    return bcube


def get_bcube_from_vertices(vertices, resolution, mip, cant_be_empty=True):
    bcube_args = []
    for dim in range(3):
        dim_min = int(min(vertices[:, dim]) // resolution[dim])
        dim_max = int(max(vertices[:, dim]) // resolution[dim]) + 1
        bcube_args.append(dim_min)
        bcube_args.append(dim_max)

    bcube = BoundingCube(*bcube_args, mip)
    if cant_be_empty and bcube.area() * bcube.z_size() == 0:
        raise Exception(
            "Attempted creation of an empty bounding "
            "when 'cant_be_empty' flag is set to True"
        )

    return bcube


class BoundingCube:
    def __init__(self, xs, xe, ys, ye, zs, ze, mip):
        self.m0_x = (None, None)
        self.m0_y = (None, None)
        self.z = (None, None)
        self.reset_coords(xs, xe, ys, ye, zs, ze, mip=mip)

    # TODO
    # def contains(self, other):
    # def insets(self, other, mip):

    def get_bounding_pts(self):
        return (self.m0_x[0], self.m0_y[0], self.z[0]), (
            self.m0_x[1],
            self.m0_y[1],
            self.z[1],
        )

    def contains(self, other):
        if self.m0_y[1] < other.m0_y[1]:
            return False
        if self.m0_x[1] < other.m0_x[1]:
            return False
        if self.z[1] < other.z[1]:
            return False

        if other.m0_x[0] < self.m0_x[0]:
            return False
        if other.m0_y[0] < self.m0_y[0]:
            return False
        if other.z[0] < self.z[0]:
            return False

        return True

    # TODO: delete?
    def intersects(self, other):
        assert type(other) == type(self)
        if other.m0_x[1] < self.m0_x[0]:
            return False
        if other.m0_y[1] < self.m0_y[0]:
            return False
        if self.m0_x[1] < other.m0_x[0]:
            return False
        if self.m0_y[1] < other.m0_y[0]:
            return False
        if self.z[1] < other.z[0]:
            return False
        if other.z[1] < self.z[0]:
            return False
        return True

    def reset_coords(
        self, xs=None, xe=None, ys=None, ye=None, zs=None, ze=None, mip=0, in_place=True
    ):

        if in_place:
            target_cube = self
        else:
            target_cube = copy.deepcopy(self)
        scale_factor = 2 ** mip
        if xs is not None:
            target_cube.m0_x = (int(xs * scale_factor), target_cube.m0_x[1])
        if xe is not None:
            target_cube.m0_x = (target_cube.m0_x[0], int(xe * scale_factor))

        if ys is not None:
            target_cube.m0_y = (int(ys * scale_factor), target_cube.m0_y[1])
        if ye is not None:
            target_cube.m0_y = (target_cube.m0_y[0], int(ye * scale_factor))

        if zs is not None:
            target_cube.z = (int(zs), target_cube.z[1])
        if ze is not None:
            target_cube.z = (target_cube.z[0], int(ze))

        return target_cube

    def get_offset(self, mip=0):
        scale_factor = 2 ** mip
        return (
            self.m0_x[0] / scale_factor + self.x_size(mip=0) / 2 / scale_factor,
            self.m0_y[0] / scale_factor + self.y_size(mip=0) / 2 / scale_factor,
        )

    def x_range(self, mip):
        scale_factor = 2 ** mip
        xl = int(round((self.m0_x[1] - self.m0_x[0]) / scale_factor))
        xs = floor(self.m0_x[0] / scale_factor)
        return [xs, xs + xl]

    def y_range(self, mip):
        scale_factor = 2 ** mip
        yl = int(round((self.m0_y[1] - self.m0_y[0]) / scale_factor))
        ys = floor(self.m0_y[0] / scale_factor)
        return [ys, ys + yl]

    def z_range(self):
        return list(copy.deepcopy(self.z))

    def area(self, mip=0):
        x_size = self.x_size(mip)
        y_size = self.y_size(mip)
        return x_size * y_size

    def x_size(self, mip):
        x_range = self.x_range(mip)
        return int(x_range[1] - x_range[0])

    def y_size(self, mip):
        y_range = self.y_range(mip)
        return int(y_range[1] - y_range[0])

    def z_size(self):
        return int(self.z[1] - self.z[0])

    def minpt(self, mip=0):
        return np.array([self.x_range(mip)[0], self.y_range(mip)[0], self.z_range()[0]])

    def maxpt(self, mip=0):
        return np.array([self.x_range(mip)[1], self.y_range(mip)[1], self.z_range()[1]])

    def to_filename(self, mip=0):
        minpt = self.minpt(mip)
        maxpt = self.maxpt(mip)
        return "_".join((str(minpt[i]) + "-" + str(maxpt[i]) for i in range(3)))

    @property
    def size(self, mip=0):
        return self.x_size(mip=mip), self.y_size(mip=mip), self.z_size()

    def crop(self, crop_xy, mip):
        scale_factor = 2 ** mip
        m0_crop_xy = crop_xy * scale_factor
        self.set_m0(
            self.m0_x[0] + m0_crop_xy,
            self.m0_x[1] - m0_crop_xy,
            self.m0_y[0] + m0_crop_xy,
            self.m0_y[1] - m0_crop_xy,
        )

    def uncrop(self, crop_xy, mip):
        """Uncrop the bounding box by crop_xy at given MIP level"""
        scale_factor = 2 ** mip
        m0_crop_xy = crop_xy * scale_factor
        result = self.clone()
        result.reset_coords(
            xs=self.m0_x[0] - m0_crop_xy,
            xe=self.m0_x[1] + m0_crop_xy,
            ys=self.m0_y[0] - m0_crop_xy,
            ye=self.m0_y[1] + m0_crop_xy,
            mip=0,
        )
        return result

    def zeros(self, mip):
        return np.zeros(
            (self.x_size(mip), self.y_size(mip), self.z_size()), dtype=np.float32
        )

    def x_res_displacement(self, d_pixels, mip):
        disp_prop = d_pixels / self.x_size(mip=0)
        result = np.full(
            (self.x_size(mip), self.y_size(mip)), disp_prop, dtype=np.float32
        )
        return result

    def y_res_displacement(self, d_pixels, mip):
        disp_prop = d_pixels / self.y_size(mip=0)
        result = np.full(
            (self.x_size(mip), self.y_size(mip)), disp_prop, dtype=np.float32
        )
        return result

    def spoof_x_y_residual(self, x_d, y_d, mip, crop_amount=0):
        x_res = crop(self.x_res_displacement(x_d, mip=mip), crop_amount)
        y_res = crop(self.y_res_displacement(y_d, mip=mip), crop_amount)
        result = np.stack((x_res, y_res), axis=2)
        result = np.expand_dims(result, 0)
        return result

    def __eq__(self, x):
        if isinstance(x, BoundingCube):
            return (self.m0_x == x.m0_x) and (self.m0_y == x.m0_y) and (self.z == x.z)
        return False

    def __str__(self, mip=0):
        return "[MIP {}] {}, {}, {}".format(
            mip, self.x_range(mip), self.y_range(mip), self.z_range()
        )

    def __repr__(self):
        return self.__str__(mip=0)

    def translate_v1(self, x=0, y=0, z=0, mip=0):
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert isinstance(z, int)

        scale_factor = 2 ** mip

        return BoundingCube(
            xs=self.m0_x[0] + x,
            xe=self.m0_x[1] + x,
            ys=self.m0_y[0] + y,
            ye=self.m0_y[1] + y,
            zs=self.z[0] + z,
            ze=self.z[1] + z,
            mip=mip,
        )

    def clone(self):
        return copy.deepcopy(self)

    def copy(self):
        return self.clone()

    def translate(self, z_offset=0, x_offset=0, y_offset=0, mip=0):
        x_range = self.x_range(mip=mip)
        y_range = self.y_range(mip=mip)
        z_range = self.z_range()
        return BoundingCube(
            xs=x_range[0] + x_offset,
            xe=x_range[1] + x_offset,
            ys=y_range[0] + y_offset,
            ye=y_range[1] + y_offset,
            zs=z_range[0] + z_offset,
            ze=z_range[1] + z_offset,
            mip=mip,
        )

    def to_slices(self, zs, ze=None, mip=0):
        x_range = self.x_range(mip=mip)
        y_range = self.y_range(mip=mip)
        return slice(*x_range), slice(*y_range), slice(*self.z)
