Set Up Walkthrough Data Path 
============================

All data read and written in this tutorial will be in the `Precomputed <https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/README.md>`_ data format and using 
`cloud-volume <https://github.com/seung-lab/cloud-volume>`_ library. 

To run the walkthrough, you need to specify the storage location where the results and intermediary data will be written. 
Provided walkthrough commands expect the location to be specified by the ``CORGIE_WALKTHROUGH_PATH`` environment variable.
``CORGIE_WALKTHROUGH_PATH`` can point either to a local directory or to cloud storage, as described bellow. 


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
for path formatting options. ``cloud-volume`` datasets stored in cloud storage can be visualized in Neuroglancer. The cloud storage bucket may need special configuration in order for the data to be accessible through Neuroglancer. Please refer to `Neuroglancer Documentation <https://github.com/google/neuroglancer>`_ for configuration details.


