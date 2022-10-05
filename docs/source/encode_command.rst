.. code-block:: bash 

	corgie apply-processor \
        --src_layer_spec '{"path": "'${CORGIE_WALKTHROUGH_PATH}'/img/img_norm"}' \
        --dst_layer_spec '{
           "type": "img",
           "path": "'${CORGIE_WALKTHROUGH_PATH}'/mip7_enc", 
           "args": {"dtype": "float32"}
           }' \
        --processor_spec '{
            "ApplyModel": {
                "params": {"path": "gs://corgie_package/models/encoders/encoder_256_2048nm"}
            }
        }' \
        --processor_mip 7 \
        --device "cuda" \
        --start_coord "100000, 100000, 17000" \
        --end_coord "150000, 150000, 17001" \
        --verbose \
        --chunk_xy 1024 --chunk_z 1
