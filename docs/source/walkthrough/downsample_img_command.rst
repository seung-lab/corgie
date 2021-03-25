
.. code-block:: bash 

   corgie downsample \
   --src_layer_spec '{
      "path": "gs://corgie/demo/my_first_stack/img/unaligned"
      }' \
   --mip_start 6 --mip_end 8 \
   --start_coord "150000, 150000, 17000" \
   --end_coord "200000, 200000, 17010" \
   --chunk_xy 1024



