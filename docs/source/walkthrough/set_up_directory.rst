Set Up Walkthrough Data Path 
============================

All data read and written in this tutorial will be in the `cloud-volume <https://github.com/seung-lab/cloud-volume>`_ format. 
You will need to specify the storage location where the results of this walkthrough will be written. 
Provided walkthrough commands expect the location to be specified by the ``CORGIE_WALKTHROUGH_PATH`` environment variable.


Using Local Directory
---------------------
To use a local directory, set the ``CORGIE_WALKTHROUGH_PATH`` environment variable to the chosen directory path with ``file://`` 
protocol prefix. To visualize ``cloud-volume`` datasets stored in your local directory, use 
`cloud-volume built-in viewing tools <https://github.com/seung-lab/cloud-volume#viewing-a-precomputed-volume-on-disk>`_.
Make sure that the specified directory exists. Example setup can be done through:

.. code:: bash

    export CORGIE_DIR=${HOME}/corgie_data/walkthrough
    mkdir -p $CORGIE_DIR
    export CORGIE_WALKTHROUGH_PATH=file://${CORGIE_DIR}



Using Cloud Storage 
-------------------
To use cloud storage, set the ``CORGIE_WALKTHROUGH_PATH`` to the cloud path. Refer to `cloud-volume Documentation <https://github.com/seung-lab/cloud-volume>`_
for path formatting options. ``cloud-volume`` datasets stored in cloud storage can be visualized in Neuroglancer the storage bucket is configured correctly.
For cloud storage configuration details, please refer to `Neuroglancer Documentation <https://github.com/google/neuroglancer>`_


