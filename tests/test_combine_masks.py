import pytest
import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec

from corgie.data_backends import str_to_backend

import shutil
import os

from corgie.boundingcube import BoundingCube
from corgie.mipless_cloudvolume import MiplessCloudVolume
from corgie.stack import Stack
from corgie.cli.combine_masks import CombineMasksTask


def delete_layer(path):
    if os.path.exists(path):
        shutil.rmtree(path)


"""
Detect consecutive masks
       0 1 2 3 4 5
INPUT    X X   X
OUTPUT   X X
"""


def make_test_dataset(path):
    mip = 0
    volume_size = Vec(1, 1, 6)
    chunk_size = Vec(1, 1, 1)
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
    I = MiplessCloudVolume(path=path, info=info, overwrite=True)
    # Make image with
    img = np.zeros(volume_size, dtype=np.uint8)
    img[:, :, 1] = 1
    img[:, :, 2] = 1
    img[:, :, 4] = 1
    I[mip][:, :, :] = img
    # view
    # ref[mip][:, :, :].viewer()


def test_combine_masks():
    src_path = "file:///tmp/cloudvolume/combine_mask_test/src"
    dst_path = "file:///tmp/cloudvolume/combine_mask_test/mask/dst"
    make_test_dataset(path=src_path)
    src_stack = Stack()
    backend = str_to_backend("cv")
    layer = backend.create_layer(
        path=src_path,
        layer_type="mask",
        name="M",
        reference=None,
    )
    src_stack.add_layer(layer)
    dst_layer = src_stack.create_unattached_sublayer(
        name="dst",
        layer_type="mask",
        custom_folder="file:///tmp/cloudvolume/combine_mask_test",
    )
    exp = {
        "inputs": [
            {
                "inputs": [
                    {"weight": 1, "key": "M", "offset": -1},
                    {"weight": 1, "key": "M", "offset": 0},
                ],
                "threshold": 1,
            },
            {
                "inputs": [
                    {"weight": 1, "key": "M", "offset": 0},
                    {"weight": 1, "key": "M", "offset": 1},
                ],
                "threshold": 1,
            },
        ],
        "threshold": 0,
    }
    bcube = BoundingCube(0, 1, 0, 1, 0, 16, mip=0)
    task = CombineMasksTask(
        src_stack=src_stack, exp=exp, dst_layer=dst_layer, mip=0, bcube=bcube, pad=0
    )
    task.execute()
    D = MiplessCloudVolume(path=dst_path)
    O = np.zeros((1, 1, 6), dtype=np.uint8)
    O[:, :, 1] = 1
    O[:, :, 2] = 1
    assert np.all(D[0][:][:, :, :, 0] == O)
    delete_layer(dst_path[7:])
    delete_layer(src_path[7:])
