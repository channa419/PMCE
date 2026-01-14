#!/bin/bash
# bash scripts/train.sh
# ================================================================MiniImageNet================================================================
python method/train.py \
    --backbone swin \
    --mode clip \
    --text_type caption \
    --feat-size 768 \
    --semantic-size 512 \
    --sem_kind imgsem 



python method/train.py \
    --backbone resnet \
    --mode clip \
    --text_type caption \
    --feat-size 640 \
    --semantic-size 512 \
    --sem_kind imgsem  
# ================================================================CIFAR_FS================================================================
python method/train.py \
    --dataset CIFAR-FS \
    --backbone swin \
    --mode clip \
    --text_type caption \
    --feat-size 768 \
    --semantic-size 512 \
    --sem_kind imgsem  

python method/train.py \
    --dataset CIFAR-FS \
    --backbone resnet \
    --mode clip \
    --text_type caption \
    --feat-size 640 \
    --semantic-size 512 \
    --sem_kind imgsem


# ================================================================FC100================================================================
python method/train.py \
    --dataset FC100 \
    --backbone swin \
    --mode clip \
    --text_type caption \
    --feat-size 768 \
    --semantic-size 512 \
    --sem_kind imgsem

python method/train.py \
    --dataset FC100 \
    --backbone resnet \
    --mode clip \
    --text_type caption \
    --feat-size 640 \
    --semantic-size 512 \
    --sem_kind imgsem


# ================================================================TieredImageNet================================================================
python method/train.py \
    --dataset TieredImageNet \
    --backbone swin \
    --mode clip \
    --text_type caption \
    --feat-size 768 \
    --semantic-size 512 \
    --sem_kind imgsem

python method/train.py \
    --dataset TieredImageNet \
    --backbone resnet \
    --mode clip \
    --text_type caption \
    --feat-size 640 \
    --semantic-size 512 \
    --sem_kind imgsem


