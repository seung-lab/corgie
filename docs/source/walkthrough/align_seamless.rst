.. code-block:: bash 

	corgie align-block \
	--src_layer_spec '{"path": "'${CORGIE_WALKTHROUGH_PATH}'/img/img_norm"}' \
	--dst_folder ${CORGIE_WALKTHROUGH_PATH}/aligned_seamless \
	--start_coord "100000, 100000, 17000" \
	--end_coord "150000, 150000, 17010" \
	--render_chunk_xy 2048 \
	--chunk_xy 1024 \
	--processor_spec '{"ApplyModel": {
        "params": {"path": "gs://corgie_package/models/MICrONS_aligner_MIP7"}
    }}' \
	--processor_mip 7 \
	--device cuda \
    --suffix aligned


