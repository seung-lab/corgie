.. code-block:: bash 

   corgie normalize \
   --src_layer_spec '{
      "path": "gs://corgie/demo/my_first_stack/img/unaligned",
         "name": "img"
         }' \
   --src_layer_spec '{
      "path":"gs://corgie/demo/my_first_stack/mask/fold_mask", 
         "type": "mask",
         "name": "fold_mask"
      }' \
      --src_layer_spec '{
         "path":"gs://corgie/demo/my_first_stack/img/unaligned", 
         "args": {"binarization": ["eq", 0.0]},
         "type": "mask",
         "name": "black_mask"
      }' \
   --dst_folder gs://corgie/demo/my_first_stack \
   --stats_mip 7 --mip_start 6 --mip_end 8 \
   --start_coord "150000, 150000, 17000" \
   --end_coord "200000, 200000, 17010" \
   --chunk_xy 2048 \
   --suffix normalized \
   --recompute_stats


