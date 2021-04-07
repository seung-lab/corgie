from corgie.stack import Stack
from corgie.log import logger as corgie_logger
from corgie.argparsers import create_layer_from_dict, create_layer_from_spec
import json

def spec_to_stack(spec, prefix, layers):
    """Create Stack by filtering a dict of layers

    Args:
        spec (dict): job_specs including src_img, src_mask, etc.
        prefix (str): src/tgt/dst
        layers (dict): int-indexed dict of layers

    job_spec will include index of layer to be used for job
    """
    stack = Stack()
    for suffix in ['img', 'mask', 'field']:
        spec_key = '{}_{}'.format(prefix, suffix)
        if spec_key in spec.keys():
            layer_id = str(spec[spec_key])
            layer = layers[layer_id]
            layer.name = suffix
            stack.add_layer(layer)
    return stack

def spec_to_layer_dict_readonly(layer_specs):
    """Create dict of layers from a corgie spec indexed by unique id

    These layers will be readonly

    Args:
        layer_specs (dict): layer specs indexed by unique id
        spec_key (str): src/tgt
    """
    layers = {}
    for k, s in layer_specs.items():
        corgie_logger.info(f'Creating layer no. {k}')
        layers[k] = create_layer_from_dict(s, readonly=True)
    return layers

def spec_to_layer_dict_overwrite(layer_specs, reference_layer, default_type):
    """Create dict of layers from a corgie spec indexed by unique id

    These layers will be of type overwrite

    Args:
        layer_specs (dict): layer specs indexed by unique id
        reference_layer (layer)
        default_type (str): e.g. img, field
    """
    layers = {}
    for k, s in layer_specs.items():
        layers[k] = create_layer_from_spec(json.dumps(s),
                                        default_type=default_type,
                                        readonly=False, 
                                        reference=reference_layer, 
                                        overwrite=True)
    return layers
