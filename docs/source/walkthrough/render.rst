Render High Resolution Images
=============================

In the previous step, we produced a MIP7 alignment of the cutout. In this step, we will apply the MIP7 field from that step to
MIP6 non-normalized images:

.. include:: render_command.rst

.. note::

    The ``"data_mip": 7`` in the field layer specification is necessary to use the field at the correct resolution.
    Without the data MIP being explicitly specified, the default behavior is to use identical field and image MIPs.  
    An alternative to providing the data MIP would be to use ``corgie upsample`` to upsample the aligned field
    from MIP7 to MIP6.
    


