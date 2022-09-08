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

All datasets read and written in this tutorial will be in the `cloud-volume <https://github.com/seung-lab/cloud-volume>`_ format.

Setting up the Destination Path
-------------------------------

You will need to specify the storage location folder for the intermediate results and final results.
Provided walkthrough commands expect the location to be specified by the ``CORGIE_WALKTHROUGH_PATH`` environment variable.


Local Directory
~~~~~~~~~~~~~~~

To use a local directory, set the ``CORGIE_WALKTHROUGH_PATH`` environment variable to the chosen directory path with ``file://`` 
protocol prefix. Make sure that the specified directory exists. For example:

.. code:: bash

    export CORGIE_DIR=${HOME}/corgie_data/walkthrough
    mkdir -p $CORGIE_DIR
    export CORGIE_WALKTHROUGH_PATH=file://${CORGIE_DIR}

To visualize ``cloud-volume`` datasets stored in your local directory, use 
`cloud-volume built-in viewing tools <https://github.com/seung-lab/cloud-volume#viewing-a-precomputed-volume-on-disk>`_.

Cloud Storage
~~~~~~~~~~~~~

To use cloud storage, set the ``CORGIE_WALKTHROUGH_PATH`` to the cloud path. Refer to `cloud-volume Documentation <https://github.com/seung-lab/cloud-volume>`_
for path formatting options.

``cloud-volume`` datasets stored in cloud storage can be visualized in Neuroglancer the storage bucket is configured correctly.
For cloud storage configuration details, please refer to `Neuroglancer Documentation <https://github.com/google/neuroglancer>`_

Preparing your stack
--------------------
.. toctree::
    :glob:

    walkthrough/copy
    walkthrough/downsample
    walkthrough/normalize

Aligning prepared stack
-----------------------
.. toctree::
    :glob:

    walkthrough/align

Next Steps
----------

.. toctree::
    :glob:

    walkthrough/next_steps
