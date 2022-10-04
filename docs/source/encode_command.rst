.. code-block:: bash 

	corgie apply-processor \
        --src_layer_spec '{
           "path": "gs://corgie_package/fafb/v15_z1000_2000/img/img"
           }' \
        --src_layer_spec '{
           "type": "img",
           "path": "'$CORGIE_WALKTHROUGH_PATH/encodings/fafb_mip3_enc'" \
           "args": {"dtype": "float32"}
           }' \
        --processor_spec '{
            "ApplyModel": {
                "params": {"path": "gs://corgie_package/models/FAFB_encoder_16_32nm"}
            }
        }'
        --processor_mip 3 \
        --start_coord "150000, 50000, 1000" \
        --end_coord "200000, 100000, 1002" \
        --chunk_xy 1024 --chunk_z 1

