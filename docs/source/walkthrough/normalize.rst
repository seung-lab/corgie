Normalize
^^^^^^^^^

The last data preparation step is normalizing the images.
In many cases alignment procedures perform better when each input section has `0.0` mean and `1.0` variance. 
It is best to perform normalization using statistics from the whole image, as normalizing each chunk independently can 
produce significant border artifacts. This is one of the reasons of why normalization is performed during data preparation
instead of being performed during alignment.


Ideally, non-tissue pixels such image defects and plastic would be excluded from statistics calculation.
For this reason, ``corgie noramlize`` command takes in an arbitrary number of mask layers as input: 

.. include:: normalize_command.rst


This command will normalize the the image layer, while excluding any pixels that belong to folds and excluding any ``0``-valued pixels, as indicated by a 
mask obtained by applying ``binarization: ["eq", 0,0]`` to the source image. The output will be written to ``${CORGIE_WALKTHROUGH_PATH}/img/img_{suffix}``, where ``--suffix=norm`` by default.

Reference expected output can be visualized in the ``Unaligned Cutout Normalized Image`` the following `Neuroglancer Link
<https://neuromancer-seung-import.appspot.com/#!%7B%22layers%22:%5B%7B%22source%22:%22precomputed://gs://corgie_package/walkthrough/example_stack/img/unaligned%22%2C%22crossSectionRenderScale%22:0.25%2C%22type%22:%22image%22%2C%22blend%22:%22default%22%2C%22shaderControls%22:%7B%7D%2C%22name%22:%22Unaligned%20Full%20Stack%20Image%22%7D%2C%7B%22source%22:%22precomputed://gs://corgie_package/walkthrough/example_stack/mask/fold_mask%22%2C%22crossSectionRenderScale%22:1.862645149230957e-9%2C%22type%22:%22segmentation%22%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22Unaligned%20Full%20Stack%20Fold%20Mask%22%2C%22visible%22:false%7D%2C%7B%22source%22:%22precomputed://gs://corgie_package/walkthrough/expected_output/img/unaligned%22%2C%22type%22:%22image%22%2C%22blend%22:%22default%22%2C%22shaderControls%22:%7B%7D%2C%22name%22:%22Unaligned%20Cutout%20Image%22%2C%22visible%22:false%7D%2C%7B%22source%22:%22precomputed://gs://corgie_package/walkthrough/expected_output/mask/fold_mask%22%2C%22type%22:%22segmentation%22%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22Unaligned%20Cutout%20Fold%20Mask%22%2C%22visible%22:false%7D%2C%7B%22source%22:%22precomputed://gs://corgie_package/walkthrough/expected_output/img/img_norm%22%2C%22type%22:%22image%22%2C%22blend%22:%22default%22%2C%22shader%22:%22void%20main%28%29%20%7B%5Cn%20%20emitGrayscale%280.5%20+%200.2%20%2A%20toNormalized%28getDataValue%28%29%29%29%3B%5Cn%7D%5Cn%22%2C%22shaderControls%22:%7B%7D%2C%22name%22:%22Unaligned%20Cutout%20Normalized%20Image%22%2C%22visible%22:false%7D%2C%7B%22source%22:%22precomputed://gs://corgie_package/walkthrough/expected_output/aligned_blockmatch/img/img_aligned%22%2C%22type%22:%22image%22%2C%22blend%22:%22default%22%2C%22shader%22:%22void%20main%28%29%20%7B%5Cn%20%20emitGrayscale%280.5%20+%200.2%20%2A%20toNormalized%28getDataValue%28%29%29%29%3B%5Cn%7D%5Cn%22%2C%22shaderControls%22:%7B%7D%2C%22name%22:%22Blockmatch%20Aligned%20Cutout%20Normalized%20Image%22%2C%22visible%22:false%7D%2C%7B%22source%22:%22precomputed://gs://corgie_package/walkthrough/expected_output/aligned_seamless/img/img_aligned%22%2C%22crossSectionRenderScale%22:5.960464477539063e-8%2C%22type%22:%22image%22%2C%22blend%22:%22default%22%2C%22shader%22:%22void%20main%28%29%20%7B%5Cn%20%20emitGrayscale%280.5%20+%200.2%20%2A%20toNormalized%28getDataValue%28%29%29%29%3B%5Cn%7D%5Cn%22%2C%22shaderControls%22:%7B%7D%2C%22name%22:%22SEAMLeSS%20Aligned%20Cutout%20Normalized%20Image%22%2C%22visible%22:false%7D%5D%2C%22navigation%22:%7B%22pose%22:%7B%22position%22:%7B%22voxelSize%22:%5B4%2C4%2C40%5D%2C%22voxelCoordinates%22:%5B198298.625%2C193354.828125%2C17000%5D%7D%7D%2C%22zoomFactor%22:528.604717707434%7D%2C%22selectedLayer%22:%7B%22layer%22:%22SEAMLeSS%20Aligned%20Cutout%20Normalized%20Image%22%2C%22visible%22:true%7D%2C%22layout%22:%22xy%22%7D>`_. 

To learn more about ``normalize`` command, please refer to :ref:`normalize command documentation <normalize command>`.

