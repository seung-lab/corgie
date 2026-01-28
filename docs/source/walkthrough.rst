Walkthrough
===========

This walkthrough demonstrates alignment process with ``corgie``.
In this walkthrough, you will align a demo stack stored in a publicly accessible GCS bucket.
| The example stack consists of one mask layer and one image layer. The data range of the stack is 
| Y: ``150000 - 200000``, 
| Y: ``150000-200000``, 
| Z: ``170000-17050``.
You can visualize the demo stack using this `Neuroglancer link <https://bit.ly/2X44Q2F>`_. 

This walkthrough is divided into two stages: stack preparation and alignment. During the stack 
preparation stage, you will make a copy of the stack cutout, downsample the cutout to a coarser
resolution, and normalize the image data. After the stack preparation stage, you will run an 
alignment command.


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
