Walkthrough
===========

This walkthrough demonstrates alignment process with ``corgie``.
In this walkthrough, you will align a demo stack stored in a publicly accessible GCS bucket.
The example stack consists of one mask layer and one image layer. The data range of the stack is 
X: ``150000-200000``, Y: ``150000-200000``, Z: ``170000-17050``.

You can visualize the demo stack using this `Neuroglancer link
<https://neuromancer-seung-import.appspot.com/#!%7B%22layers%22:%5B%7B%22source%22:%22precomputed://gs://corgie_package/walkthrough/example_stack/img/unaligned%22%2C%22crossSectionRenderScale%22:0.25%2C%22type%22:%22image%22%2C%22blend%22:%22default%22%2C%22shaderControls%22:%7B%7D%2C%22name%22:%22Unaligned%20Full%20Stack%20Image%22%7D%2C%7B%22source%22:%22precomputed://gs://corgie_package/walkthrough/example_stack/mask/fold_mask%22%2C%22crossSectionRenderScale%22:1.862645149230957e-9%2C%22type%22:%22segmentation%22%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22Unaligned%20Full%20Stack%20Fold%20Mask%22%2C%22visible%22:false%7D%2C%7B%22source%22:%22precomputed://gs://corgie_package/walkthrough/expected_output/img/unaligned%22%2C%22type%22:%22image%22%2C%22blend%22:%22default%22%2C%22shaderControls%22:%7B%7D%2C%22name%22:%22Unaligned%20Cutout%20Image%22%2C%22visible%22:false%7D%2C%7B%22source%22:%22precomputed://gs://corgie_package/walkthrough/expected_output/mask/fold_mask%22%2C%22type%22:%22segmentation%22%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22Unaligned%20Cutout%20Fold%20Mask%22%2C%22visible%22:false%7D%2C%7B%22source%22:%22precomputed://gs://corgie_package/walkthrough/expected_output/img/img_norm%22%2C%22type%22:%22image%22%2C%22blend%22:%22default%22%2C%22shader%22:%22void%20main%28%29%20%7B%5Cn%20%20emitGrayscale%280.5%20+%200.2%20%2A%20toNormalized%28getDataValue%28%29%29%29%3B%5Cn%7D%5Cn%22%2C%22shaderControls%22:%7B%7D%2C%22name%22:%22Unaligned%20Cutout%20Normalized%20Image%22%2C%22visible%22:false%7D%2C%7B%22source%22:%22precomputed://gs://corgie_package/walkthrough/expected_output/aligned_blockmatch/img/img_aligned%22%2C%22type%22:%22image%22%2C%22blend%22:%22default%22%2C%22shader%22:%22void%20main%28%29%20%7B%5Cn%20%20emitGrayscale%280.5%20+%200.2%20%2A%20toNormalized%28getDataValue%28%29%29%29%3B%5Cn%7D%5Cn%22%2C%22shaderControls%22:%7B%7D%2C%22name%22:%22Blockmatch%20Aligned%20Cutout%20Normalized%20Image%22%2C%22visible%22:false%7D%2C%7B%22source%22:%22precomputed://gs://corgie_package/walkthrough/expected_output/aligned_seamless/img/img_aligned%22%2C%22crossSectionRenderScale%22:5.960464477539063e-8%2C%22type%22:%22image%22%2C%22blend%22:%22default%22%2C%22shader%22:%22void%20main%28%29%20%7B%5Cn%20%20emitGrayscale%280.5%20+%200.2%20%2A%20toNormalized%28getDataValue%28%29%29%29%3B%5Cn%7D%5Cn%22%2C%22shaderControls%22:%7B%7D%2C%22name%22:%22SEAMLeSS%20Aligned%20Cutout%20Normalized%20Image%22%2C%22visible%22:false%7D%5D%2C%22navigation%22:%7B%22pose%22:%7B%22position%22:%7B%22voxelSize%22:%5B4%2C4%2C40%5D%2C%22voxelCoordinates%22:%5B198298.625%2C193354.828125%2C17000%5D%7D%7D%2C%22zoomFactor%22:528.604717707434%7D%2C%22selectedLayer%22:%7B%22layer%22:%22SEAMLeSS%20Aligned%20Cutout%20Normalized%20Image%22%2C%22visible%22:true%7D%2C%22layout%22:%22xy%22%7D>`_. 

This walkthrough is divided into two stages: stack preparation and alignment. During the stack 
preparation stage, you will make a copy of a cutout from the stack, downsample the cutout to a coarser
resolution, and normalize the image data. After the stack preparation stage, you will run align the 
cutout with two different alignment approaches.


.. toctree:: 
    :glob:
    :caption: Contents

    walkthrough/set_up_directory
    walkthrough/copy
    walkthrough/downsample
    walkthrough/normalize
    walkthrough/align
    walkthrough/render
    walkthrough/next_steps
