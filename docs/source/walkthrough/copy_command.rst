
.. code-block:: bash 

	corgie copy \
	--src_layer_spec '{
	   "name": "unaligned",
	   "path": "gs://corgie/demo/example_stack/img/unaligned"
	   }' \
	--src_layer_spec '{
	   "name": "fold_mask",
	   "type": "mask",
	   "path": "gs://corgie/demo/example_stack/mask/fold_mask"
	   }' \
        --dst_folder "gs://corgie/demo/my_first_stack" \
	--mip 6 \
	--start_coord "150000, 150000, 17000" \
	--end_coord "200000, 200000, 17010" \
	--chunk_xy 1024 --chunk_z 1
