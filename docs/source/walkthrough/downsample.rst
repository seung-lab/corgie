Downsample
^^^^^^^^^^
Next, we want to downsample the MIP6 data that we copied for easier visualization. By downsampling we meed producing coarser data at higher MIP levels. In this tutorial, we will downsample the layer from MIP6 to MIP6. In order to do that, rught the following commands:

Normalize image:

.. include:: downsample_img_command.rst

Normalize fold mask:

.. include:: downsample_mask_command.rst

Most of the parameters of ``downample`` command are same as with the ``copy`` command we used earlier -- we specify start and end coordinates, chunk size, and the source layer. Unlike ``copy`` command, we do not have to provice the destination parameter. When ``--dst_layer_spec`` is not specified, the downsampled data will be written to the source layer. 

Downsampling for the mask and the image layer have to be done separately, becuase image and mask data require different downsampling strategies. To learn more about downsampling strategies, please refer to TOD:downsampling_strategies.

To learn more about ``downsample`` command, please refer to :ref:`downsample command documentation <downsample command>`.


