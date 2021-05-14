#!/bin/bash

for file in ./test_specs/*
do
    echo python generate_gt ${file}
    python generate_gt.py ${file}
done
