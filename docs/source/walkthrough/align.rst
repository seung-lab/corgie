Align
=====

Block Matching
--------------

First, let's align the stack using conventional block matching image pair alignment method.
This is done by providing an alignment processor that implements block matching to the ``corgie align-block`` command:

.. include:: align_blockmatch.rst

The ``--processor_spec`` specifies an image pair alignment method, and ``--processor_mip`` specifies what resolution to apply it to. You can can change the ``tile_size``, ``tile_step``, ``max_disp`` and ``r_delta`` parameters and see how it affects your result. To avoid rewriting data from previous runs, use different ``--dst_folder`` and/or ``--suffix``. 

To learn more about the meaning of ``tile_size``, ``tile_step``, ``max_disp`` and ``r_delta`` parameters, please refer to `this link <https://imagej.net/Elastic_Alignment_and_Montage>`_.

Reference expected output can be visualized with the following `Neuroglancer Link <>`_.

Alignment in the resulting stack is generally improved, but the discontinuous defects are not corrected.

SEAMLeSS
--------
To correct discontinuous defects, let's retry the alignment, but this time using SEAMLeSS image pair alignment method:

.. include:: align_seamless.rst

Reference expected output can be visualized with the following `Neuroglancer Link <>`_. 

The numerous discontinuous defects present in the stack are now corrected when viewed at MIP7 resolution.

To learn more about ``align-block`` command, please refer to :ref:`align-block command documentation <align-block command>`.

