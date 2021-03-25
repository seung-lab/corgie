Align
^^^^^

After we copied, downsampled, and normalized the image stack, we are ready to align. This time we will be aligning using a pre-build ``corgie`` Block-Matching model: 

.. include:: align_block_command.rst

The ``--processor_spec`` specifies with "processor" to use for alignment, and ``--processor_mip`` specifies what resolution to apply it to. You can can change the ``tile_size``, ``tile_step``, ``max_disp`` and ``r_delta`` parameters and see how it affects your result. To avoid rewriting data from previous runs, use different ``--dst_folder`` and/or ``--suffix``. 

To learn more about the meaning of ``tile_size``, ``tile_step``, ``max_disp`` and ``r_delta`` parameters, please refer to `this link <https://imagej.net/Elastic_Alignment_and_Montage>`_.

To learn more about processor specification, please refer to TODO:processor_spec.

To learn more about ``align-block`` command, please refer to :ref:`align-block command documentation <align-block command>`.
