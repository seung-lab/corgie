.. code-block:: bash 

	corgie render \
        --src_layer_spec '{
           "name": "img",
           "path": "'${CORGIE_WALKTHROUGH_PATH}'/img/unaligned"
           }' \
        --src_layer_spec '{
           "name": "fold_mask",
           "type": "mask",
           "path": "'${CORGIE_WALKTHROUGH_PATH}'/mask/fold_mask"
           }' \
        --src_layer_spec '{
           "type": "field",
           "path": "'${CORGIE_WALKTHROUGH_PATH}'/aligned_seamless/field/field_aligned",
           "args": {"data_mip": 7}
        }' \
        --dst_folder $CORGIE_WALKTHROUGH_PATH/aligned_seamless \
        --mip 6 \
        --start_coord "150000, 150000, 17000" \
        --end_coord "200000, 200000, 17010" \
        --chunk_xy 1024 \
        --chunk_z 1 \
        --suffix warped 

