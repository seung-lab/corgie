# Stack class
# Stack of sections
# defined by:
# 3d bounding box
# cv
# section is a stack with thickness == 1
import six
import os
import copy
import torch

from corgie import exceptions, helpers
from corgie.log import logger as corgie_logger
from corgie.layers import str_to_layer_type

class StackBase:
    def __init__(self, name=None):
        self.name = name
        self.layers = {}
        self.reference_layer = None

    def add_layer(self, layer):
        if layer.name is None:
            raise exceptions.UnnamedLayerException(layer, f"Layer name "
                    f"needs to be set for it to be added to {self.name} stack.")
        #if layer.name in self.layers:
        #    raise exceptions.ArgumentError(layer, f"Layer with name "
        #            f"'{layer.name}' added twice to '{self.name}' stack.")
        if self.reference_layer is None:
            self.reference_layer = layer
        self.layers[layer.name] = layer

    def remove_layer(self, layer_name):
        del self.layers[layer_name]

    def __contains__(self, m):
        return m in self.layers.keys()

    def __getitem__(self, m):
        return self.layers[m]

    def __len__(self):
        return len(self.layers)

    def read_data_dict(self, **index):
        result = {}
        for l in layers:
            result[l.name] = l.read(**index)
        raise NotImplementedError

    def write_data_dict(self, data_dict):
        raise NotImplementedError

    def get_layers(self):
        return list(self.layers.values())

    def get_layers_of_type(self, type_names):
        if isinstance(type_names, str):
            type_names = [type_names]

        types = tuple(str_to_layer_type(n) for n in type_names)

        result = []
        for k, v in six.iteritems(self.layers):
            if isinstance(v, types):
                result.append(v)

        return result

    def get_layer_types(self):
        layer_types = set()
        for k, v in six.iteritems(self.layers):
            layer_types.add(v.get_layer_type())

        return list(layer_types)


class Stack(StackBase):
    def __init__(self, name=None, layer_list=[], folder=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.folder = folder

        for l in layer_list:
            self.add_layer(l)

    def create_sublayer(self, name, layer_type, suffix='', reference=None, **kwargs):
        if self.folder is None:
            raise exceptions.CorgieException("Stack must have 'folder' field set "
                    "before sublayers can be created")

        if self.reference_layer is None and reference is None:
            raise exceptions.CorgieException("Stack must either have at least one layer "
                    "or reference layer must be provided for sublayer creation")

        if reference is None:
            reference = self.reference_layer

        path = os.path.join(self.folder, layer_type, f"{name}{suffix}")
        l = reference.backend.create_layer(path=path, layer_type=layer_type,
                name=name, reference=reference, **kwargs)
        self.add_layer(l)
        return l

    def read_data_dict(self, bcube, mip, translation_adjuster=None, stack_name=None, add_prefix=True,
            translation=None):

        data_dict = {}

        if translation is None:
            translation = helpers.Translation(0, 0)

        if stack_name is None:
            stack_name == self.name

        if stack_name is None or not add_prefix:
            name_prefix = ""
        else:
            name_prefix = f"{stack_name}_"


        agg_field = None
        field_layers = self.get_layers_of_type("field")
        # Assume that the last field is the final one
        if len(field_layers) > 0:
            l = field_layers[-1]
            global_name = "{}{}".format(name_prefix, l.name)
            this_field = l.read(bcube=bcube, mip=mip)
            data_dict[global_name] = this_field
            agg_field = this_field

        '''for l in field_layers:
            global_name = "{}{}".format(name_prefix, l.name)
            this_field = l.read(bcube=bcube, mip=mip)
            data_dict[global_name] = this_field
            agg_field = this_field
            if agg_field is None:
                agg_field = this_field
            else:
                agg_field = this_field.field().from_pixels()(agg_field.field().from_pixels()).pixels()'''

        assert (f"{name_prefix}agg_field" not in data_dict)
        data_dict[f"{name_prefix}agg_field"] = agg_field

        if translation_adjuster is not None:
            src_field_trans = translation_adjuster(agg_field)
            translation.x += src_field_trans.x
            translation.y += src_field_trans.y

        #if translation.x != 0 or translation.y != 0:
        #import pdb; pdb.set_trace()
        final_bcube = copy.deepcopy(bcube)
        final_bcube = final_bcube.translate(
                x_offset=translation.y,
                y_offset=translation.x,
                mip=mip)

        for l in self.get_layers_of_type(["mask", "img"]):
            global_name = f"{name_prefix}{l.name}"
            data_dict[global_name] = l.read(bcube=final_bcube, mip=mip)
        return translation, data_dict

    def z_range(self):
        return self.bcube.z_range()

    def cutout(self):
        raise NotImplementedError

def create_stack_from_reference(reference_stack, folder, name, types=None, suffix='', **kwargs):
    result = Stack(name=name, folder=folder)
    if types is None:
        layers = reference_stack.get_layers()
    else:
        layers = reference_stack.get_layers_of_type(types)

    for l in layers:
        result.create_sublayer(name=l.name, layer_type=l.get_layer_type(), suffix=suffix,
                reference=l, dtype=l.get_data_type(), **kwargs)
    return result

class FieldSet():
    """Collection of Field layers to handle composition
    """
    def __init__(self, layers=[]):
        self.layers = layers

    def append(self, layer):
        self.layers.append(layer)

    def get_field(self, layer, bcube, mip):
        """Get field, adjusted by distance

        Args:
            layer (Layer)
            bcube (BoundingCube)
            mip (int)
            # dist (float)
        
        Returns:
            TorchField # adjusted (blurred/attenuated) by distance
        """
        # TODO: add ability to get blurred field using trilinear interpolation
        # c = min(max(dist / self.decay_dist, 0.), 1.)
        corgie_logger.debug(f'get_field')
        corgie_logger.debug(f'\tlayer={layer}')
        corgie_logger.debug(f'\tbcube={bcube}')
        corgie_logger.debug(f'\tmip={mip}')
        f = layer.read(bcube=bcube, mip=mip).field_()
        return f # * c

    def read(self, bcube, z_list, mip):
        """Compute composition of fields indexed by bcube & z_list

        This takes a list of fields, [f_0, f_1, ..., f_n],
        and composes them to get
        f_0 ⚬ f_1 ⚬ ... ⚬ f_n ~= f_0(f_1(...(f_n)))
        """
        if isinstance(z_list, int):
            z_list = [z_list] * len(self.layers)
        assert(len(z_list) == len(self.layers))

        src_z = z_list[-1]
        z = z_list[0]
        layer = self.layers[0]
        abcube = bcube.reset_coords(zs=z, ze=z+1, in_place=False)
        agg_field = self.get_field(layer=layer,
                                    bcube=abcube,
                                    mip=mip)
                                    # dist=src_z - z)
        for z, layer in zip(z_list[1:], self.layers[1:]):
            trans = helpers.percentile_trans_adjuster(agg_field)
            corgie_logger.debug(f'{trans}')
            abcube = abcube.reset_coords(zs=z, ze=z+1, in_place=True)
            abcube = abcube.translate(x_offset=trans.y,
                                      y_offset=trans.x,
                                      mip=mip)
            trans = trans.to_tensor()
            agg_field -= trans
            agg_field = agg_field.from_pixels()
            this_field = self.get_field(layer=layer,
                                        bcube=abcube,
                                        mip=mip)
                                        # dist=src_z - z)
            this_field = this_field.from_pixels()
            agg_field = agg_field(this_field)
            agg_field = agg_field.pixels()
            agg_field += trans
        return agg_field