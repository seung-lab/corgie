import copy

import torch
from tqdm import tqdm

from corgie import constants, exceptions

from corgie.log import logger as corgie_logger
from corgie.boundingcube import BoundingCube
from corgie.layers.base import register_layer_type, BaseLayerType
from corgie import helpers


def get_extra_interpolate_parameters():
    # torch.nn.function.interpolate complains if this
    # argument is not provided, but it doesn't exist in older versions
    from packaging import version

    if version.parse(torch.__version__) <= version.parse("1.4.0"):
        return {}
    return {"recompute_scale_factor": False}

class BCubeIterator:
    """
    Computable list. Supports iterating and slice operations.
    Useful for very large lists that would exhaust RAM.
    Can be split using the slice operator to parallelize.
    """
    def __init__(
        self, 
        x_range, y_range, z_range,
        sx, sy, sz, 
        chunk_xy_step, chunk_z_step,
        mip
    ):
        self.start = 0
        self.end = (sx * sy * sz)
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.chunk_xy_step = chunk_xy_step
        self.chunk_z_step = chunk_z_step
        self.mip = mip
    def __len__(self):
        return self.end - self.start
    def __iter__(self):
        for i in tqdm(range(self.start, self.end)):
            yield self.get_i(i)
    def __getitem__(self, slc):
        if isinstance(slc, int):
            return self.get_i(slc)
        itr = copy.deepcopy(self)
        itr.start = max(self.start + slc.start, self.start)
        itr.end = min(self.start + slc.stop, self.end)
        return itr
    def get_i(self, i):
        x, y, z = self.to_coord(i)
        return self.get(x,y,z)
    def get(self, x, y, z):
        return BoundingCube(
            x, x + self.chunk_xy_step, 
            y, y + self.chunk_xy_step, 
            z, z + self.chunk_z_step, 
            mip=self.mip
        )
    def to_coord(self, i):
        """Convert an index into a grid coordinate."""
        if i >= len(self) or i < 0:
            raise ValueError(f"{i} out of bounds.")

        sxy = self.sx * self.sy
        z = i // sxy
        y = (i - (z * sxy)) // self.sx
        x = i - self.sx * (y + z * self.sy)
        x = self.x_range[0] + x * self.chunk_xy_step
        y = self.y_range[0] + y * self.chunk_xy_step
        z = self.z_range[0] + z * self.chunk_z_step
        return (x,y,z)


class VolumetricLayer(BaseLayerType):
    def __init__(self, data_mip=None, **kwargs):
        super().__init__(**kwargs)
        self.data_mip = data_mip

    def read(self, bcube, mip, **kwargs):
        indexed_bcube = self.indexing_scheme(bcube, mip, kwargs)
        if self.data_mip is not None:
            if mip <= self.data_mip:
                result_data_mip = super().read(
                    bcube=indexed_bcube, mip=self.data_mip, **kwargs
                )
                result = result_data_mip
                for _ in range(mip, self.data_mip):
                    result = self.get_upsampler()(result)
            elif mip > self.data_mip:
                # TODO: consider restricting MIP difference to prevent memory blow-up
                result_data_mip = super().read(
                    bcube=indexed_bcube, mip=self.data_mip, **kwargs
                )
                result = result_data_mip
                for _ in range(self.data_mip, mip):
                    result = self.get_downsampler()(result)
            else:
                # TODO: consider restricting MIP difference to prevent memory blow-up
                result_data_mip = super().read(
                    bcube=indexed_bcube, mip=self.data_mip, **kwargs
                )
                result = result_data_mip
                for _ in range(self.data_mip, mip):
                    result = self.get_downsampler()(result)
        else:
            result = super().read(bcube=indexed_bcube, mip=mip, **kwargs)

        return result

    def write(self, data_tens, bcube, mip, **kwargs):
        indexed_bcube = self.indexing_scheme(bcube, mip, kwargs)
        super().write(data_tens=data_tens, bcube=indexed_bcube, mip=mip, **kwargs)

    def indexing_scheme(self, bcube, mip, kwargs):
        return bcube

    def break_bcube_into_chunks(
        self,
        bcube,
        chunk_xy,
        chunk_z,
        mip,
        chunk_xy_step=None,
        chunk_z_step=None,
        **kwargs
    ):
        """Default breaking up of a bcube into smaller bcubes (chunks).
        Returns a list of chunks
        Args:
           bcube: BoundingBox for region to be broken into chunks
           chunk_size: tuple for dimensions of chunk that bbox will be broken into
           mip: int for MIP level at which chunk_xy is dspecified
        """
        indexed_bcube = self.indexing_scheme(bcube, mip, kwargs)

        x_range = indexed_bcube.x_range(mip=mip)
        y_range = indexed_bcube.y_range(mip=mip)
        z_range = indexed_bcube.z_range()

        if chunk_xy_step is None:
            chunk_xy_step = chunk_xy
        if chunk_z_step is None:
            chunk_z_step = chunk_z

        # round up
        sx = (x_range[1] - x_range[0] + (chunk_xy_step - 1)) // chunk_xy_step
        sy = (y_range[1] - y_range[0] + (chunk_xy_step - 1)) // chunk_xy_step
        sz = (z_range[1] - z_range[0] + (chunk_z_step - 1)) // chunk_z_step
        
        return BCubeIterator(
            x_range, y_range, z_range,
            sx, sy, sz, 
            chunk_xy_step, chunk_z_step,
            mip
        )

@register_layer_type("img")
class ImgLayer(VolumetricLayer):
    def __init__(self, *args, num_channels=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels = num_channels

    def get_downsampler(self):
        def downsampler(data_tens):
            return torch.nn.functional.interpolate(
                data_tens.float(),
                mode="bilinear",
                scale_factor=1 / 2,
                align_corners=False,
                **get_extra_interpolate_parameters()
            )

        return downsampler

    def get_upsampler(self):
        def upsampler(data_tens):
            return torch.nn.functional.interpolate(
                data_tens.float(),
                mode="bilinear",
                scale_factor=2.0,
                align_corners=False,
                **get_extra_interpolate_parameters()
            )

        return upsampler

    def get_num_channels(self, *args, **kwargs):
        return self.num_channels

    def get_default_data_type(self):
        return "uint8"


@register_layer_type("segmentation")
class SegmentationLayer(VolumetricLayer):
    def __init__(self, *args, num_channels=1, **kwargs):
        if num_channels != 1:
            raise exceptions.ArgumentError(
                "Segmentation layer 'num_channels'",
                "Segmentation layer must have 1 channels. 'num_channels' provided: {}".format(
                    num_channels
                ),
            )
        super().__init__(*args, **kwargs)

    def read(self, dtype=None, **kwargs):
        return self.read_backend(transpose=False, **kwargs).squeeze()

    def get_downsampler(self):
        def downsampler(data_tens):
            downs_data = torch.nn.functional.interpolate(
                data_tens.float(),
                mode="nearest",
                scale_factor=1 / 2,
                align_corners=False,
                **get_extra_interpolate_parameters()
            )
            return downs_data

        return downsampler

    def get_upsampler(self):
        def upsampler(data_tens):
            ups_data = torch.nn.functional.interpolate(
                data_tens.float(),
                mode="nearest",
                scale_factor=2.0,
                align_corners=False,
                **get_extra_interpolate_parameters()
            )
            return ups_data

        return upsampler

    def get_num_channels(self, *args, **kwargs):
        return 1

    def get_default_data_type(self):
        return "uint64"


@register_layer_type("field")
class FieldLayer(VolumetricLayer):
    """Residuals are specified at a relative resolution based on MIP
    """

    def __init__(self, *args, num_channels=2, **kwargs):
        if num_channels != 2:
            raise exceptions.ArgumentError(
                "Field layer 'num_channels'",
                "Field layer must have 2 channels. 'num_channels' provided: {}".format(
                    num_channels
                ),
            )
        super().__init__(*args, **kwargs)

    def get_downsampler(self):
        def downsampler(data_tens):

            downs_data = torch.nn.functional.interpolate(
                data_tens.float(),
                mode="bilinear",
                scale_factor=1 / 2,
                align_corners=False,
                **get_extra_interpolate_parameters()
            )
            return downs_data * 0.5

        return downsampler

    def get_upsampler(self):
        def upsampler(data_tens):
            ups_data = torch.nn.functional.interpolate(
                data_tens.float(),
                mode="bilinear",
                scale_factor=2.0,
                align_corners=False,
                **get_extra_interpolate_parameters()
            )
            return ups_data * 2

        return upsampler

    def get_num_channels(self, *args, **kwargs):
        return 2

    def get_default_data_type(self):
        return "float32"


@register_layer_type("mask")
class MaskLayer(VolumetricLayer):
    def __init__(self, binarization=None, num_channels=1, **kwargs):
        self.binarizer = helpers.Binarizer(binarization)
        if num_channels != 1:
            raise exceptions.ArgumentError(
                "Mask layer 'num_channels'",
                "Mask layer must have 1 channels. 'num_channels' provided: {}".format(
                    num_channels
                ),
            )
        super().__init__(**kwargs)

    def read(self, **kwargs):
        data_tens = super().read(**kwargs)
        data_bin = self.binarizer(data_tens)
        return data_bin

    def get_downsampler(self):
        def downsampler(data_tens):
            return torch.nn.functional.max_pool2d(data_tens.float(), kernel_size=2)

        return downsampler

    def get_upsampler(self):
        def upsampler(data_tens):
            return torch.nn.functional.interpolate(
                data_tens.float(),
                mode="nearest",
                scale_factor=2.0,
                **get_extra_interpolate_parameters()
            )

        return upsampler

    def get_num_channels(self, *args, **kwargs):
        return 1

    def get_default_data_type(self):
        return "uint8"


@register_layer_type("section_value")
class SectionValueLayer(VolumetricLayer):
    def __init__(self, *args, num_channels=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels = num_channels

    # TODO: insert custom indexing here.

    def get_num_channels(self, *args, **kwargs):
        return 1

    def indexing_scheme(self, bcube, mip, kwargs):
        new_bcube = copy.deepcopy(bcube)
        if "channel_start" in kwargs and "channel_end" in kwargs:
            channel_start = kwargs["channel_start"]
            channel_end = kwargs["channel_end"]
            del kwargs["channel_start"], kwargs["channel_end"]
        else:
            channel_start = 0
            channel_end = self.num_channels

        new_bcube.reset_coords(channel_start, channel_end, 0, 1, mip=mip)
        return new_bcube

    def supports_voxel_offset(self):
        return False

    def supports_chunking(self):
        return False

    def get_default_data_type(self):
        return "float32"


@register_layer_type("fixed_field")
class FixedFieldLayer(FieldLayer):
    """Residuals are specified at a fixed resolution, regardless of MIP.
    For example, a field at MIP2 may have residuals specified in MIP0 
    pixels.

    NOTE: It is not recommended to create new fields based on this
    class. This class was intended to support existing fields created
    by legacy code. 
    """

    def __init__(self, *args, fixed_mip=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_mip = fixed_mip

    def get_downsampler(self):
        def downsampler(data_tens):

            downs_data = torch.nn.functional.interpolate(
                data_tens.float(),
                mode="bilinear",
                scale_factor=1 / 2,
                align_corners=False,
                **get_extra_interpolate_parameters()
            )
            return downs_data

        return downsampler

    def get_upsampler(self):
        def upsampler(data_tens):
            ups_data = torch.nn.functional.interpolate(
                data_tens.float(),
                mode="bilinear",
                scale_factor=2.0,
                align_corners=False,
                **get_extra_interpolate_parameters()
            )
            return ups_data

        return upsampler

    def read(self, bcube, mip, **kwargs):
        return super().read(bcube, mip, **kwargs) / (2 ** (mip - self.fixed_mip))

    # def write(self, data_tens, bcube, mip, **kwargs):
    #     super().write(data_tens=data_tens*(2**self.fixed_mip),
    #                   bcube=indexed_bcube,
    #                   mip=mip,
    #                   **kwargs)


@register_layer_type("float_tensor")
class FloatTensorLayer(VolumetricLayer):
    """Generic 4D float tensors"""
    def __init__(self, *args, num_channels=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels = num_channels

    def get_downsampler(self):
        def downsampler(data_tens):
            return torch.nn.functional.interpolate(
                data_tens.float(),
                mode="bilinear",
                scale_factor=1 / 2,
                align_corners=False,
                **get_extra_interpolate_parameters()
            )

        return downsampler

    def get_upsampler(self):
        def upsampler(data_tens):
            return torch.nn.functional.interpolate(
                data_tens.float(),
                mode="bilinear",
                scale_factor=2.0,
                align_corners=False,
                **get_extra_interpolate_parameters()
            )

        return upsampler

    def get_num_channels(self, *args, **kwargs):
        return self.num_channels

    def get_default_data_type(self):
        return "float32"
