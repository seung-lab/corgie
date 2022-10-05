Supplying Custom Image Pair Alignment Methods 
=============================================

``corgie`` allows users to plug in their own image pair alignment methods. 

Custom models are plugged into ``corgie`` by specifying the path in ``--processor_spec`` specification. 
The path must point to a folder, which must contain an  ``__init__.py`` and ``create.py`` python source files.
``create.py`` file must implement a ``create`` function which returns an object with ``torch.nn.Module`` subtype.

The following is an example ``create.py`` for the ``blockmatch`` processing model used in the walkthrough: 

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
               pred_field = block_match(
                   src_img, tgt_img, 
                   tile_size=self.tile_size,
                   tile_step=self.tile_step, 
                   max_disp=self.max_disp,
                   min_overlap_px=500, 
                   filler=0, 
                   r_delta=self.r_delta
               )

               return pred_field

    def create(**kwargs):
        return Model(**kwargs)

In this case, the actual block matching business logic is contained in a separate ``blockmatch.py`` file.
Full contents of this processor folder are available at ``gs://corgie_package/models/blockmatch``.

Arguments to the ``create`` function call can be provided through ``--processor_spec``.
``corgie`` will use the constructed model to process image pair chunks, passed in as pytorch tensors as inputs to the ``forward`` call.
Chunk data for all layers passed to the ``align`` or ``align-block`` will be passed in as keyword arguments to ``forward`` for both the source and the target image..
Source layers are passed as ``src_{layer name}`` and target layers are passed as ``tgt_{layer name}``. It is recommended for the models to take a variable number of keyword arguments (``**kwargs``), as the user might provide unanticipated layers. 

The image pair alignment model is expected to return a saturated displacement field in `torchfields <https://github.com/seung-lab/torchfields>`_ format.


.. note::

    Once loaded, the model will be cached on the worker. That means that if you update the contents of the model path, 
    some of the workers might still use a stale version of the model which is saved in their cache. To prevent this issue, 
    we recommend either restarting all of the workers when the contents of a model folder are updated, or not overwriting 
    contents of existing models and creating new models instead.  
