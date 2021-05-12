# Automated Testing

This folder has integration tests for `corgie` commands. It stores the original data and ground truth results as local cloudvolumes, which adds about 7MB to this repo.

# Running tests

## 1. Run the worker
`./run_worker.sh`


### 2. Run Tests
All tests:

`python -m pytest`

Only test one command:

`python -m pytest test_copy.py`

# Adding tests
