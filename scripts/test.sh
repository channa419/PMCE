#!/bin/bash
# bash scripts/test.sh
# ================================================================MiniImageNet================================================================

# backbone=swin
# python method/test.py --use_enhancer --shot 1 
# python method/test.py --use_enhancer --shot 5 --map_estimate_cov true --map_alpha 0.7  --map_topk 7 

# backbone=renset12
# python method/test.py --use_enhancer --shot 1 --backbone resnet --feat-size 640
# python method/test.py --use_enhancer --shot 5 --map_estimate_cov true --map_alpha 0.7 --map_topk 7 --backbone resnet --feat-size 640

# ================================================================CIFAR_FS================================================================
# backbone=swin
# python method/test.py --shot 1 --dataset CIFAR-FS --use_enhancer 
# python method/test.py --shot 5 --dataset CIFAR-FS --use_enhancer --map_estimate_cov true --map_alpha 0.7 --map_topk 7 

# backbone=renset12
# python method/test.py --shot 1 --dataset CIFAR-FS --use_enhancer --backbone resnet --feat-size 640
# python method/test.py --shot 5 --dataset CIFAR-FS --use_enhancer --backbone resnet --feat-size 640 --map_estimate_cov true --map_alpha 0.7  --map_topk 7 



# ================================================================FC100================================================================
# backbone=swin
# python method/test.py --shot 1 --dataset FC100 --use_enhancer 
# python method/test.py --shot 5 --dataset FC100 --use_enhancer --map_estimate_cov true --map_alpha 0.7 --map_topk 7 


# backbone=renset12
# python method/test.py --shot 1 --dataset FC100 --use_enhancer --backbone resnet --feat-size 640
# python method/test.py --shot 5 --dataset FC100 --use_enhancer --backbone resnet --feat-size 640 --map_estimate_cov true --map_alpha 0.7 --map_topk 7 
# ================================================================TieredImageNet================================================================
# backbone=swin
# python method/test.py --shot 1 --dataset TieredImageNet --use_enhancer 
# python method/test.py --shot 5 --dataset TieredImageNet --use_enhancer --map_estimate_cov true --map_alpha 0.7 --map_topk 7 


# backbone=renset12
# python method/test.py --shot 1 --dataset TieredImageNet --use_enhancer --backbone resnet --feat-size 640
# python method/test.py --shot 5 --dataset TieredImageNet --use_enhancer --backbone resnet --feat-size 640 --map_estimate_cov true --map_alpha 0.7 --map_topk 7 
