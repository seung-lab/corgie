Layers
======

Layer Types
---------------

Image Layer 
^^^^^^^^^^^

Field Layer
^^^^^^^^^^^

Mask Layer
^^^^^^^^^^

Section Value Layer
^^^^^^^^^^^^^^^^^^^

Layer Backends
---------------



Specifying Layer as Input
-------------------------

Format: JSON *string*::

   Optional keys: 
    "type": str from ["img", "mask", "field"] ; 
    "name": str, DEFAULT -- same as layer type.
            [required] if given more than 1 layer of the same type ; 
    "data_backend": str from ["cv"] ; 
    "layer_args": a dictionary with additional layer parameters,
                  eg binarization scheme for masks data_mip_ranges, etc; 
   Required keys: 
    "path": path to the layer data;
   


