from corgie.layers.base import get_layer_types, str_to_layer_type

# when adding a new layer type, include it here
from corgie.layers.volumetric_layers import (
    FieldLayer,
    ImgLayer,
    MaskLayer,
    SectionValueLayer,
    SegmentationLayer,
    FixedFieldLayer,
    FloatTensorLayer,
)

# also update the default layer type if you want to
DEFAULT_LAYER_TYPE = "img"
