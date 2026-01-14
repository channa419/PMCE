#!/bin/bash
# bash scripts/compute_center.sh

# ================================================================MiniImageNet================================================================
python compute_center.py  \
    --dataset MiniImageNet \
    --backbone swin \
     

python compute_center.py  \
    --dataset MiniImageNet \
    --backbone resnet \
# ================================================================CIFAR-FS================================================================
python compute_center.py  \
    --dataset CIFAR-FS \
    --backbone swin \
     

python compute_center.py  \
    --dataset CIFAR-FS \
    --backbone resnet \

# ================================================================FC100================================================================
python compute_center.py  \
    --dataset FC100 \
    --backbone swin \

python compute_center.py  \
    --dataset FC100 \
    --backbone resnet \

# ================================================================TieredImageNet================================================================
python compute_center.py  \
    --dataset TieredImageNet \
    --backbone swin \semantic
