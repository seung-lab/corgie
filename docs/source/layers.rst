Specifying Input Layers
=======================


Layers are passed as input to commands as JSON strings with the following format::

   Optional keys: 
    "type": str from ["img", "mask", "field"] ; 
    "name": str, DEFAULT -- same as layer type.
            [required] if given more than 1 layer of the same type ; 
    "args": a dictionary with additional layer parameters,
                  eg binarization scheme for masks data_mip_ranges, etc; 
   Required keys: 
    "path": path to the layer data;
   


