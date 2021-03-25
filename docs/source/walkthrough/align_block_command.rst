.. code-block:: bash 

   corgie align-block \
   --src_layer_spec '{"path": "gs://corgie/demo/my_first_stack/img/img_normalized"}' \
   --dst_folder gs://corgie/demo/my_first_stack/aligned \
   --start_coord "100000, 100000, 17000" \
   --end_coord "150000, 150000, 170010" \
   --chunk_xy 2048 \
   --suffix run_x0 \
   --processor_spec '{"ApplyModel": {
      "params": {
         "path": "gs://corgie/models/blockmatch",
         "tile_size": 128,
         "tile_step": 64,
         "max_disp": 48,
         "r_delta": 1.3
     }}}' \
   --processor_mip 7 


