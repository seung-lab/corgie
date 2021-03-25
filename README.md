# COnnectomics Registration Generalizable Inference Engine (corgie)

Welcome to corgie! corgie is a command line tool built for registration of very large 3D volumes.

# Installation

```
pip install corgie
```

# Example 

```
corgie copy \
--src_layer_spec '{
   "name": "unaligned",
   "path": "https://s3-hpcrc.rc.princeton.edu/minnie65-phase3-em/unaligned"
   }' \
--src_layer_spec '{
   "name": "large_folds",
   "type": "mask",
   "path": "gs://seunglab_minnie_phase3/alignment/unaligned_fold_lengths/threshold_350"
   }' \
--dst_folder "gs://corgie/demo/my_first_stack" \
--start_coord "150000, 150000, 17000" \
--end_coord "250000, 250000, 17020" \
--mip 6 \
--chunk_xy 1024 --chunk_z 1
```

# Documentation
[Link](https://corgie.readthedocs.io/en/latest/)
