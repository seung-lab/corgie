import pytest
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec
from corgie.mipless_cloudvolume import MiplessCloudVolume


def create_dummy_mipless_cloudvolume():
    mip = 0
    volume_size = Vec(128, 64, 30)
    chunk_size = Vec(64, 64, 1)
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="image",
        data_type="uint8",
        encoding="raw",
        resolution=[1, 1, 1],
        voxel_offset=[0, 0, 0],
        chunk_size=chunk_size,
        volume_size=volume_size,
    )
    return MiplessCloudVolume(
        path="file:///tmp/corgie/test", info=info, overwrite=False
    )


def test_set_param():
    cv = create_dummy_mipless_cloudvolume()
    cv.set_param("cache", True)
    assert cv.cv_params["cache"]
    cv.set_param("cache", False)
    assert not cv.cv_params["cache"]

