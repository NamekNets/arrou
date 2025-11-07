#!/bin/bash

echo "activating environment"
source /mnt/Shared-Storage/sid/miniconda3/bin/activate
conda activate nameknets

echo "running unlearning_wmdp.py"
CUDA_VISIBLE_DEVICES=0 python unlearning_wmdp.py --task cyber --method rmu

conda deactivate
