# PMCE: Probabilistic Multi-Granularity Semantics with Caption-Guided Enhancement for Few-Shot Learning




---

# 1 Prerequisites


### 1.1 Install Dependencies
```bash
cd PMCE
# Install required packages
pip install -r requirements.txt
```

### 1.2 Dataset Setup
**Download Datasets**: Prepare the following few-shot benchmarks:
   - MiniImageNet
   - TieredImageNet
   - CIFAR-FS
   - FC100


**Storage**: Save the datasets to your desired storage path. 

**Configuration**: Ensure you update the corresponding data paths in the configuration files or command-line arguments.




### 1.3 Vision–Language Models Preparation

This project relies on two key pretrained models:

- **BLIP** (for image caption generation)  
  Model: `Salesforce/blip-image-captioning-base`

- **CLIP** (for visual/text semantic encoding)  
  Model: `openai/clip-vit-base-patch32`

#### Model Usage Modes

#### Online Mode

If your environment has internet access, the `transformers` library will automatically download and cache the models.

- #### Configuration
- Modify the code in `build_caption_semantic.py`
- Set `local_files_only=False` in both `load_blip` and `load_clip`

- #### Run
```bash
python build_caption_semantic.py \
  --blip_dir "Salesforce/blip-image-captioning-base" \
  --clip_dir "openai/clip-vit-base-patch32"
  ```
- #### Downloaded models will be cached at:
```bash
~/.cache/huggingface/hub
```


#### Offline Mode
If your server cannot access the internet, please download all model files in advance.
- ####  Download link
- [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)


- #### Configuration
Keep the code unchanged.Download all files and place them in a local directory.Ensure 
``` code
local_files_only=True
``` 




# 2 Pretrained Weights and Semantic Files

The pre-trained weights for the **Swin-Tiny** backbone (trained on MiniImageNet), the semantic features, and the pre-computed **Base Priors** are already included in this repository. 

# 3 Start
Please execute the following commands in sequence.
### 3.1 Compute Base-Class Priors
Compute class means for each dataset（Alternatively, you can use the files we have already provided. ）:
``` code
python compute_center.py \
  --dataset MiniImageNet/TieredImageNet/CIFAR-FS/FC100 \
  --backbone resnet/swin
  ``` 

### 3.2 Prepare Semantic Features

Generate semantic features for train/val/test splits

``` code
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

  ``` 

### 3.3 Train the Enhancer
You may either train the enhancer or use the provided pretrained weights.
``` code
python method/train.py \
  --backbone swin \
  --mode clip \
  --text_type caption \
  --feat-size 768 \
  --semantic-size 512 \
  --sem_kind imgsem
  ``` 

### 3.4 Evaluation
1-shot Evaluation
``` code
python method/test.py \
  --use_enhancer \
  --shot 1 \
  --backbone resnet/swin \
  --dataset MiniImageNet/TieredImageNet/CIFAR-FS/FC100 \
  --map_estimate_cov true \
  --map_alpha 0.33 \
  --map_topk 7
  ``` 
5-shot Evaluation
``` code
python method/test.py \
  --use_enhancer \
  --shot 5 \
  --backbone resnet/swin \
  --dataset MiniImageNet/TieredImageNet/CIFAR-FS/FC100 \
  --map_estimate_cov true \
  --map_alpha 0.7 \
  --map_topk 7
  ``` 

