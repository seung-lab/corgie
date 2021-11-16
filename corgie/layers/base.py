import torch
import numpy as np

from corgie import constants, exceptions
from corgie import helpers


STR_TO_LTYPE_DICT  = dict()


def register_layer_type(layer_type_name):
    def register_layer_fn(layer_type):
        STR_TO_LTYPE_DICT[layer_type_name] = layer_type
        def get_layer_type_name(obj):
            return layer_type_name
        layer_type.get_layer_type = get_layer_type_name
        return layer_type

    return register_layer_fn


def str_to_layer_type(s):
    global STR_TO_LTYPE_DICT
    return STR_TO_LTYPE_DICT[s]


def get_layer_types():
    return list(STR_TO_LTYPE_DICT.keys())


class BaseLayerType:
    def __init__(self, name=None, device='cpu', readonly=False, **kwargs):
        super().__init__()
        self.device = device
        self.readonly = readonly
        self.name = name

    def read(self, dtype=None, **kwargs):
        data_np = self.read_backend(**kwargs)
        # TODO: if np type is unit32, convert it to int64
        if data_np.dtype == np.uint32:
            data_np = data_np.astype(np.int64)
        data_tens = torch.as_tensor(data_np, device=kwargs.get('device', self.device))
        data_tens = helpers.cast_tensor_type(data_tens, dtype)
        return data_tens

    def write(self, data_tens, **kwargs):
        if self.readonly:
            raise Exception("Attempting to write into a readonly layer {}".format(str(self)))
        data_np = data_tens
        if torch.is_tensor(data_tens):
            data_tens = helpers.expand_to_dims(data_tens, 4)
            if data_tens.dtype == torch.float64:
                data_tens = data_tens.float()
            #if data_tens.dtype in [torch.float32, torch.float64]:
                #import pdb; pdb.set_trace()
                #if self.get_data_type() in ['uint8']:
                #    data_tens = data_tens * 255
                #if self.get_data_type() not in ['float32', 'float64', 'float']:
                #    raise Exception("Unknown conversiotn between float and int")

            data_np = data_tens.data.cpu().numpy().astype(
                    self.get_data_type()
                    )
        self.write_backend(data_np, **kwargs)


    def supports_voxel_offset(self):
        return True

    def supports_chunking(self):
        return True

    def get_downsampler(self, *kargs, **kwargs):
        raise NotImplementedError

    def get_upsampler(self, *kargs, **kwargs):
        raise NotImplementedError

    def get_num_channels(self, *kargs, **kwargs):
        raise NotImplementedError
