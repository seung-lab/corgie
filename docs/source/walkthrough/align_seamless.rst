.. code-block:: bash 

	corgie align-block \
	--src_layer_spec '{"path": "'${CORGIE_WALKTHROUGH_PATH}'/img/img_normalized"}' \
	--dst_folder ${CORGIE_WALKTHROUGH_PATH_REMOTE}/aligned_seamless \
	--start_coord "100000, 100000, 17000" \
	--end_coord "150000, 150000, 17010" \
	--chunk_xy 2048 \
	--render_chunk_xy 2048 \
	--pad 64 \
	--processor_spec '{"ApplyModel": {"params": 
			{"path": "gs://corgie/models/pyramid_m4m6m9/0_mip7in_mip9module",
			 "finetune_iter": 100, "checkpoint_name": "metric_net",
			 "finetune_sm": 30
	}}}' \
	--processor_mip 7 \
	--device cuda


