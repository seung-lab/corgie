.. code-block:: bash 

   corgie downsample \
   --src_layer_spec '{
      "path": "'${CORGIE_WALKTHROUGH_PATH}'/mask/fold_mask",
      "type": "mask"
      }' \
   --mip_start 6 --mip_end 8 \
   --start_coord "150000, 150000, 17000" \
   --end_coord "200000, 200000, 17010" \
   --chunk_xy 1024

