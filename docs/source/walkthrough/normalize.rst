Normalize
^^^^^^^^^

The last step before we start the actual alignment is normalization of images. In many cases alignment procedures perform better when each input section has `0.0` mean and `1.0` variance. However, simply normalizing all of the pixels in each section for the specified bounding cube can produce biases result when part of the section is missing or defected. For this reason, we use several masks in the normalization commad:

.. include:: normalize_command.rst

The first mask that we use is the fold mask we copied from the reference stack. The second mask is obtained by applying ``binarization: ["eq", 0,0]`` to the source image, which will mask out all the ``0`` valued pixels in the image. 

``normalize`` command works in two steps -- first it computes mean and variance for each section, and then it normalizes each section individually. The MIP at which mean and variance are calculated is specified by ``--stats_mip``. 

We also speciffy the ``--suffix`` to be used for the resulting layer -- in this case, the normalized image will be written out to ``gs://corgie/demo/my_first_stack/img/img_normalized``.

To learn more about ``normalize`` command, please refer to :ref:`normalize command documentation <normalize command>`.

