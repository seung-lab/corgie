Normalize
^^^^^^^^^

The last data preparation step involves normalizing the images.
In many cases alignment procedures perform better when each input section has `0.0` mean and `1.0` variance. 
It is best to perform normalization using statistics from the whole image, as normalizing each chunk independently can 
produce significant border artifacts. This is one of the reasons of why normalization is performed during data preparation
instead of being performed during alignment.

Another note is that ideally image defects, plastic, and other non-tissue pixels would be excluted from statistics calculation.
For this reason, ``corgie noramlize`` command takes in an arbitrary number of mask layers as input: 

.. include:: normalize_command.rst


This command will normalize the the image layer, while excluding any pixels that belong to folds and excluding any ``0``-valued pixels, as indicated by a 
mask obtained by applying ``binarization: ["eq", 0,0]`` to the source image.

To learn more about ``normalize`` command, please refer to :ref:`normalize command documentation <normalize command>`.

