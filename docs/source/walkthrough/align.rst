Align
=====

We will produce two alignment versions, one using conventional block matching image pair alignment, and another using SEAMLeSS ConvNet based approach.
Both alignments will be performed at MIP7, the resulting alignment field will be saved at MIP7 resolution to ``{dst_folder}/field/field_aligned``. 

Block Matching
--------------

First, let's align the stack using conventional block matching image pair alignment method.
This is done by providing an alignment processor that implements block matching to the ``corgie align-block`` command:

.. include:: align_blockmatch.rst

The ``--processor_spec`` specifies an image pair alignment method, and ``--processor_mip`` specifies what resolution to apply it to. You can can change the ``tile_size``, ``tile_step``, ``max_disp`` and ``r_delta`` parameters and see how it affects your result. To avoid rewriting data from previous runs, use different ``--dst_folder`` and/or ``--suffix``. 

To learn more about the meaning of ``tile_size``, ``tile_step``, ``max_disp`` and ``r_delta`` parameters, please refer to `this link <https://imagej.net/Elastic_Alignment_and_Montage>`_.

Reference expected output can be visualized in the ``Blockmatch Aligned Cutout Normalized Image`` the following `Neuroglancer Link <https://tinyurl.com/corgie-wakthrough>`_. Alignment in the resulting stack is generally improved, but the discontinuous defects are not corrected. Note that the alignment quality can be improved by finding more optimal hyperparameters.

SEAMLeSS
--------
To correct discontinuous defects, let's retry the alignment, but this time using SEAMLeSS image pair alignment method:

.. include:: align_seamless.rst


.. note::

   The command specifies usage of GPU accelerator through ``--device cuda``. If your environment does not have cuda-enabled GPU,
   please use ``--device cpu`` instead.



Reference expected output can be visualized in the ``SEAMLeSS Aligned Cutout Normalized Image`` the following `Neuroglancer Link <https://tinyurl.com/corgie-wakthrough>`_. The numerous discontinuous defects present in the stack are now corrected when viewed at MIP7 resolution.

To learn more about ``align-block`` command, please refer to :ref:`align-block command documentation <align-block command>`.

