.. code-block:: bash 

   corgie normalize \
   --src_layer_spec '{
      "path": "'${CORGIE_WALKTHROUGH_PATH}'/img/unaligned",
         "name": "img"
         }' \
   --src_layer_spec '{
      "path":"'${CORGIE_WALKTHROUGH_PATH}'/mask/fold_mask", 
         "type": "mask",
         "name": "fold_mask"
      }' \
      --src_layer_spec '{
         "path":"'${CORGIE_WALKTHROUGH_PATH}'/img/unaligned", 
         "args": {"binarization": ["eq", 0.0]},
         "type": "mask",
         "name": "black_mask"
      }' \
   --dst_folder ${CORGIE_WALKTHROUGH_PATH} \
   --stats_mip 7 --mip_start 6 --mip_end 8 \
   --start_coord "150000, 150000, 17000" \
   --end_coord "200000, 200000, 17010" \
   --chunk_xy 1024 \
   --suffix normed


