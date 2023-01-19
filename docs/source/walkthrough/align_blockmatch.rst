.. code-block:: bash 

   corgie align-block \
   --src_layer_spec '{"path": "'${CORGIE_WALKTHROUGH_PATH}'/img/img_norm"}' \
   --dst_folder ${CORGIE_WALKTHROUGH_PATH}/aligned_blockmatch \
   --start_coord "100000, 100000, 17000" \
   --end_coord "150000, 150000, 17010" \
   --chunk_xy 2048 \
   --render_chunk_xy 2048 \
   --suffix run_x0 \
   --processor_spec '{"ApplyModel": {
      "params": {
         "path": "https://storage.googleapis.com/corgie_package/models/aligners/blockmatch",
         "tile_size": 128,
         "tile_step": 64,
         "max_disp": 48,
         "r_delta": 1.3
     }}}' \
   --processor_mip 7 


