#!/bin/bash
# original_x0
corgie copy \
    --src_layer_spec '{"path": "gs://tmacrina-corgie-test/fafbv15_per_island/affine_blockmatch_combined_v1/complete_mip6"}' \
    --src_layer_spec '{"path": "gs://tmacrina-corgie-test/fafbv15_per_island/affine_blockmatch_combined_v1/folds", "type": "mask"}' \
    --dst_folder "file://./test_data/original/original_x0" \
    --start_coord "0, 0, 100" \
    --end_coord "278400, 130944, 120" \
    --chunk_xy 2048 --chunk_z 1 \
    --mip 7 \
    --force_chunk_xy 2048 \
    --force_chunk_z 1


