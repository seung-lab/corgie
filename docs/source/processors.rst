Processors
===========

``corgie`` allows users to create and use custom models. Moreover, different models can be combined with each other to enable more interesting results.

This is achieved through use of two libraries -- ``modelhouse`` and ``procspec``. 

.. _modelhouse create:

Creating Models
---------------
``modelhouse`` allows users to create and load cutom processing models, which can then be loaded by remote workers. ``modelhouse`` comes automatically with ``corgie`` installation, but you can also install it through::
   
   pip install modelhouse


To upload a new model, use::
   
   modelhouse create -s {Source model folder} -d {Destination folder where the model will reside}

Most common usage is to create and test a model locally on your machine, and then upload it to Google Storage or AWS, where it can be accessed by remote workers.

Model Requirements
^^^^^^^^^^^^^^^^^^

Each model folder must provide a ``create.py`` file in which it defines a ``create`` function, which returns a model object. This function will be called by the workers in order to construct instantiations of the model. 

The following is an example ``create.py`` for the ``blockmatch`` processing model used in the re

.. code-block:: python
 
   from blockmatch import block_match

   class Model(nn.Module):
       def __init__(self, tile_size=64, tile_step=32, max_disp=32, r_delta=1.1):
	   super().__init__()
	   self.tile_size = tile_size
	   self.tile_step = tile_step
	   self.max_disp = max_disp
	   self.r_delta = r_delta

       def forward(self, src_img, tgt_img, **kwargs):
	   pred_field = block_match(src_img, tgt_img, tile_size=self.tile_size,
				    tile_step=self.tile_step, max_disp=self.max_disp,
				    min_overlap_px=500, filler=0, r_delta=self.r_delta)

	   return pred_field

      def create(**kwargs):
	  return Model(**kwargs)


``corgie`` will pass all the layers specified by a command to the model. By default, source layers are passed as "src_{layer name} and target layers are passed as "tgt_{layer name}". It is recommended for the models to take a variable number of keyword arguments (``**kwars``), as the user might provide unanticipated layers. 


.. note::
   Once loaded, the model will be cached on the worker. That means that if you update the contents of the model path, some of the workers might still use a stale version of the model which is saved in their cache. To prevent this issue, we recommend either restarting all of the workers when the contents of a model folder are updated, or not overwriting contents of existing models and creating new models instea.  


Combine models and modify model operation
-----------------------------------------
``procspec`` is a library that provides a simple way to define processor specification. Each processor specification is a JSON string. The simplest processor which only applies one ``blockmatch`` model is defined as follows:

.. code-block:: bash 

   {
	"ApplyModel": {
	    "params": {
	       "path": "gs://corgie/models/blockmatch",
	       "tile_size": 128,
	       "tile_step": 64,
	       "max_disp": 48,
	       "r_delta": 1.3
	    }
	}
   }


Meaning of each field:

| ``ApplyModel``   -- this processor is a model
| ``params``       -- parameters necessary to create the model
| ``path``         -- path to the model (as created by :ref:`modelhouse create <modelhouse create>`)
| the rest 	 -- parameters passed to the ``blockmatch`` constructor    	


Here's an example of slightly more complex processor:

.. code-block:: bash 

   [
      {
	"ApplyModel": {
	    "output_key": "src_encoding",
	    "input_keys": {"src_img": "src_img"},
	    "params": {
	       "path": "gs://corgie/models/encoder"
	    }
	}
     },
     {
	"ApplyModel": {
	    "output_key": "tgt_encoding",
	    "input_keys": {"src_img": "tgt_img"},
	    "params": {
	       "path": "gs://corgie/models/encoder"
	    }
	}
     },
     {
	"ApplyModel": {
	    "params": {
	       "input_keys": {
		  "src_img": "src_encoding",
		  "tgt_img": "tgt_encoding"
	       },
	       "path": "gs://corgie/models/blockmatch",
	       "tile_size": 128,
	       "tile_step": 64,
	       "max_disp": 48,
	       "r_delta": 1.3
	    }
	}
     }
  ]

This is a list of ``ApplyModel`` processors. These 3 processors will be applied sequentially, and outputs of the first two processors will be used as inputs to the last processor. More documentation on complex processors will be added soon. 
