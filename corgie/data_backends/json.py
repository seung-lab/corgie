import os
import copy
import numpy as np

from cloudfiles import CloudFiles

from corgie import layers
from corgie import exceptions
from corgie.log import logger as corgie_logger

from corgie.data_backends.base import DataBackendBase, BaseLayerBackend, \
        register_backend, str_to_backend


@register_backend("json")
class JSONDataBackend(DataBackendBase):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
json_backend = str_to_backend("json")

class JSONLayerBase(BaseLayerBackend):
    """A directory with one text file per section
    """
    def __init__(self, path, backend, reference=None, overwrite=True, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.dtype = 'O'
        self.backend = backend
        self.cf = CloudFiles(self.path, progress=False)

    def __str__(self):
        return "JSON {}".format(self.path)

    def get_sublayer(self, name, layer_type=None, path=None, **kwargs):
        if path is None:
            path = os.path.join(self.path, layer_type, name)

        if layer_type is None:
            layer_type = self.get_layer_type()

        return self.backend.create_layer(path=path, reference=self,
                layer_type=layer_type, **kwargs)

    def get_filename(self, z):
        return f'{z:06d}'

    def read_backend(self, bcube, **kwargs):
        z_range = bcube.z_range()
        corgie_logger.debug(f'Read from {str(self)}, z: {z_range}')
        data = []
        for z in z_range:
            f = self.cf.get_json(self.get_filename(z))
            data.append(f)
        return data

    def write_backend(self, data, bcube, **kwargs):
        z_range = range(*bcube.z_range())
        assert(len(data) == len(z_range))
        corgie_logger.debug(f'Write to {str(self)}, z: {z_range}')
        filepaths = [self.get_filename(z) for z in z_range]
        self.cf.put_jsons(zip(filepaths, data), cache_control='no-cache')

@json_backend.register_layer_type_backend("section_value")
class JSONSectionValueLayer(JSONLayerBase, layers.SectionValueLayer):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
