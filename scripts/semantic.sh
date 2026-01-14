#!/bin/bash


# bash scripts/semantic.sh
# ================================================================miniImageNet================================================================
python build_caption_semantic.py \
  --dataset_root /path/to/your/datasets/MiniImageNet/train \
  --imagenet_index_json ./imagenet_class_index.json \
  --blip_dir /path/to/your/offline_models/blip-image-captioning-base \
  --clip_dir /path/to/your/offline_models/clip-vit-base-patch32 \
  --dataset MiniImageNet \
  --split train \
  --mode clip \
  --text_type caption \
  --output_dir ./semantic \
  --clsname_source imagenet_index 

python build_caption_semantic.py \
  --dataset_root /path/to/your/datasets/MiniImageNet/val \
  --imagenet_index_json ./imagenet_class_index.json \
  --blip_dir /path/to/your/offline_models/blip-image-captioning-base \
  --clip_dir /path/to/your/offline_models/clip-vit-base-patch32 \
  --dataset MiniImageNet \
  --split val \
  --mode clip \
  --text_type caption \
  --output_dir ./semantic \
  --clsname_source imagenet_index 

python build_caption_semantic.py \
  --dataset_root /path/to/your/datasets/MiniImageNet/test \
  --imagenet_index_json ./imagenet_class_index.json \
  --blip_dir /path/to/your/offline_models/blip-image-captioning-base \
  --clip_dir /path/to/your/offline_models/clip-vit-base-patch32 \
  --dataset MiniImageNet \
  --split test \
  --mode clip \
  --text_type caption \
  --output_dir ./semantic \
  --clsname_source imagenet_index  
# ================================================================CIFAR_FS================================================================
# python build_caption_semantic.py \
#   --dataset_root /path/to/your/cifar_fs/train \
#   --blip_dir /path/to/your/offline_models/blip-image-captioning-base \
#   --clip_dir /path/to/your/offline_models/clip-vit-base-patch32 \
#   --dataset CIFAR-FS \
#   --dataset_name cifar_fs \
#   --split train \
#   --mode clip \
#   --text_type caption \
#   --output_dir ./semantic \
#   --clsname_source folder_name  


# python build_caption_semantic.py \
#   --dataset_root /path/to/your/cifar/test \
#   --blip_dir /path/to/your/offline_models/blip-image-captioning-base \
#   --clip_dir /path/to/your/offline_models/clip-vit-base-patch32 \
#   --dataset CIFAR-FS \
#   --dataset_name cifar_fs \
#   --split test \
#   --mode clip \
#   --text_type caption \
#   --output_dir ./semantic \
#   --clsname_source folder_name  


# python build_caption_semantic.py \
#   --dataset_root /path/to/your/cifar/val \
#   --blip_dir /path/to/your/offline_models/blip-image-captioning-base \
#   --clip_dir /path/to/your/offline_models/clip-vit-base-patch32 \
#   --dataset CIFAR-FS \
#   --dataset_name cifar_fs \
#   --split val \
#   --mode clip \
#   --text_type caption \
#   --output_dir ./semantic \
#   --clsname_source folder_name  

# ================================================================FC100================================================================

# python build_caption_semantic.py \
#   --dataset_root /path/to/your/FC100/train \
#   --blip_dir /path/to/your/offline_models/blip-image-captioning-base \
#   --clip_dir /path/to/your/offline_models/clip-vit-base-patch32 \
#   --dataset FC100 \
#   --dataset_name fc100 \
#   --split train \
#   --mode clip \
#   --text_type caption \
#   --output_dir ./semantic \
#   --clsname_source folder_name  

# python build_caption_semantic.py \
#   --dataset_root /path/to/your/FC100/val \
#   --blip_dir /path/to/your/offline_models/blip-image-captioning-base \
#   --clip_dir /path/to/your/offline_models/clip-vit-base-patch32 \
#   --dataset FC100 \
#   --dataset_name fc100 \
#   --split val \
#   --mode clip \
#   --text_type caption \
#   --output_dir ./semantic \
#   --clsname_source folder_name 

# python build_caption_semantic.py \
#   --dataset_root /path/to/your/FC100/test \
#   --dataset FC100 \
#   --dataset_name fc100 \
#   --split test \
#   --mode clip \
#   --text_type caption \
#   --output_dir ./semantic \
#   --clsname_source folder_name 


# ================================================================TieredImageNet================================================================
# python build_caption_semantic.py \
#   --dataset_root /path/to/your/tiered_imagenet/train \
#   --blip_dir /path/to/your/offline_models/blip-image-captioning-base \
#   --clip_dir /path/to/your/offline_models/clip-vit-base-patch32 \
#   --dataset TieredImageNet \
#   --dataset_name tieredimagenet \
#   --split train \
#   --mode clip \
#   --text_type caption \
#   --output_dir ./semantic \
#   --clsname_source imagenet_index  

# python build_caption_semantic.py \
#   --dataset_root /path/to/your/tiered_imagenet/val \
#   --blip_dir /path/to/your/offline_models/blip-image-captioning-base \
#   --clip_dir /path/to/your/offline_models/clip-vit-base-patch32 \
#   --dataset TieredImageNet \
#   --dataset_name tieredimagenet \
#   --split val \
#   --mode clip \
#   --text_type caption \
#   --output_dir ./semantic \
#   --clsname_source imagenet_index  


# python build_caption_semantic.py \
#   --dataset_root /path/to/your/tiered_imagenet/test \
#   --blip_dir /path/to/your/offline_models/blip-image-captioning-base \
#   --clip_dir /path/to/your/offline_models/clip-vit-base-patch32 \
#   --dataset TieredImageNet \
#   --dataset_name tieredimagenet \
#   --split test \
#   --mode clip \
#   --text_type caption \
#   --output_dir ./semantic \
#   --clsname_source imagenet_index  