import json
import click

from click_option_group import optgroup

from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, str_to_layer_type, \
        DEFAULT_LAYER_TYPE
from corgie.data_backends import get_data_backends, str_to_backend, \
        DEFAULT_DATA_BACKEND
from corgie import exceptions

from corgie.stack import Stack

corgie_optgroup = optgroup.group

def corgie_option(*args, **kwargs):
    return optgroup.option(*args, show_default=True, **kwargs)

LAYER_HELP_STR = """
   [required]\n
   Format: JSON *string*. \n
   Optional keys: \n
   \t "type": str from {} ; \n
   \t "name": str, DEFAULT -- same as layer type. """\
   """**Required** if given more than 1 layer of the same type ; \n
   \t "data_backend": str from {} ; \n
   \t "layer_args": a dictionary with additional layer parameters,
   eg binarization scheme for masks data_mip_ranges, etc; \n
   Required keys: "path". \n
   """.format(get_layer_types(), get_data_backends())

   #\t "data_mip_ranges": array of [int, int] tupples specifying MIP ranges where the layer has data. DEFAULT -- all MIPs have data.\n

def create_layer_from_spec(spec_str, **kwargs):
    return create_layer_from_dict(json.loads(spec_str), **kwargs)

def create_layer_from_dict(param_dict, reference=None, caller_name=None,
        allowed_types=None, default_type=None, **kwargs):
    if default_type is None:
        default_type = DEFAULT_LAYER_TYPE
    default_param_dict = {
            "path": None,
            "name": None,
            "type": default_type,
            "data_backend": DEFAULT_DATA_BACKEND,
            "args": {},
            "readonly": False
            }
    for k in param_dict.keys():
        if k not in default_param_dict:
            raise exceptions.CorgieException(f"Unkown layer parameter '{k}'")
    params = {**default_param_dict, **param_dict}

    if params["path"] is None:
        arg_spec = '"path" key in layer specification'
        if caller_name is not None:
            arg_spec += ' of {}'.format(caller_name)
        raise exceptions.ArgumentError(arg_spec, 'not given')

    layer_path = params["path"]
    layer_type = params["type"]
    layer_args = params["args"]
    data_backend = params["data_backend"]
    corgie_logger.debug("Parsing layer path: {}".format(
        layer_path
        ))

    if allowed_types is not None and layer_type not in allowed_types:
        raise exceptions.ArgumentError("layer_type",
                f'must be of type in {allowed_types}')

    backend = str_to_backend(data_backend)
    layer = backend.create_layer(path=layer_path, layer_type=layer_type,
            reference=reference, layer_args=layer_args,
            **kwargs)

    name = params["name"]
    if name is None:
        name = layer.get_layer_type()
    layer.name = name

    return layer


def create_stack_from_spec(spec_str_list, name, reference=None, readonly=False,
        **kwargs):
    stack = None
    if len(spec_str_list) == 0:
        if reference is not None:
            stack = Stack(name=name)
            for l in reference.get_layers():
                l.readonly = readonly
                stack.add_layer(l)
    else:
        stack = Stack(name=name)
        layer_list = [create_layer_from_spec(s, readonly=readonly, **kwargs) \
                for s in spec_str_list]
        for l in layer_list:
            stack.add_layer(l)

    return stack

