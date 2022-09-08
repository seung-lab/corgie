Copy
^^^^
In order to make a copy of the portion of the stack, run the following in your command line: 

.. include:: copy_command.rst


The command specifies source data layers through ``--src_layer_spec``. 
This stack has two layers, so ``--src_layer_spec`` flag is used twice. 
Next, ``--dst_folder`` specifies the destination folder where all of the source layers will be copied. 

Other than the layer sources and destination paths, the command also specifies the region of interest through ``--start_coord`` and ``--end_coord``. 
Both start and end coordinates are provided as 3 comma separated integers -- X, Y and Z coordinates. 
The start and end coordinates form a bounding cube, and all of the source data within this bounding cube will be copied over to the destination. 
In this example, we will copy all the data in X range ``150000-200000``, Y range
``150000-200000``, and Z range ``17000-17010``. 
Notice that we are not copying the whole example stack, just the first 10 sections of it. 
``--mip`` specifies what MIP of the data will be copied, which in this case is 6. 

.. note::
   By default, X and Y coordinates are considered to be at MIP0. You can change this default behavior by passing `--coord_mip` parameter

Lastly, we need to specify the chunking parameters. 
``corgie`` is designed to deal with large datasets that cannot fit in memory, and so it operates on large images in smaller chunks. 
``--chunk_xy`` with ``--chunk_z`` define the size of each chunk. 
In this example, our stack will be copied in chunks of size ``1024x1024x1`` at MIP6. 

To learn more about ``copy`` command, please refer to :ref:`copy command documentation <copy command>`.

.. note::
   Unlike start and end coordinates, chunk size is considered to be provided at the MIP used for processing, which is MIP6 in this example.
