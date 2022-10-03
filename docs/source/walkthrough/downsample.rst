Downsample
^^^^^^^^^^

In the previous step, we copied a cutout from the demo stack at MIP6 resolution.
In this step, we will downsample the copied MIP6 data to MIP7 and MIP8 resolutions.
This is done through the following commands:

Downsample image:

.. include:: downsample_img_command.rst

Downsample fold mask:

.. include:: downsample_mask_command.rst

Note that unlike with ``copy`` command, we the destination path parameter is not mandatory and will default to the source layer path. 

Downsampling for the mask and the image layer have to be done separately, because images are downsampled with average pooling strategy while masks are downsampled
with max pooling strategy. You can refer to the following `article <https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/>`_ to 
learn more about the difference between different pooling methods.

To learn more about ``downsample`` command, please refer to :ref:`downsample command documentation <downsample command>`.


