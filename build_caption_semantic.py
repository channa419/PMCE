import os
import json
import argparse
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import torch.nn.functional as F   
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPTokenizer,
    CLIPModel,
)


def load_blip(blip_dir: str, device: torch.device):
    """
    Load BLIP caption model offline.
    """
    processor = BlipProcessor.from_pretrained(blip_dir, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(
        blip_dir, local_files_only=True
    ).to(device)
    model.eval()
    return processor, model


def load_clip(clip_dir: str, device: torch.device):
    """
    Load CLIP text encoder offline.
    """
    tokenizer = CLIPTokenizer.from_pretrained(clip_dir, local_files_only=True)
    model = CLIPModel.from_pretrained(clip_dir, local_files_only=True).to(device)
    model.eval()
    return tokenizer, model


def load_wnid_to_name(imagenet_index_json: str):
    """
    Load wnid -> class name mapping.
    e.g. "n01532829" -> "house finch"
    """
    with open(imagenet_index_json, "r", encoding="utf-8") as f:
        idx_dict = json.load(f)

    wnid_to_name = {}
    for _, (wnid, name_str) in idx_dict.items():
        # name_str might be "house_finch", replace _ with space
        wnid_to_name[wnid] = name_str.replace("_", " ")
    return wnid_to_name


@torch.no_grad()
def generate_caption(image: Image.Image,
                     processor: BlipProcessor,
                     blip_model: BlipForConditionalGeneration,
                     device: torch.device,
                     max_new_tokens: int = 30) -> str:
    """
    Generate English caption for a single image.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    output_ids = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    caption = caption.strip()
    return caption


@torch.no_grad()
def encode_text_clip(texts,
                     tokenizer: CLIPTokenizer,
                     clip_model: CLIPModel,
                     device: torch.device) -> torch.Tensor:
    """
    Encode batch of texts to 512-dim vectors using CLIP.
    Returns normalized tensor (B, 512).
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    text_embeds = clip_model.get_text_features(**inputs)  # (B, 512)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return text_embeds.cpu()


def main():
    parser = argparse.ArgumentParser()

    # ===== Basic Paths =====
    parser.add_argument(
        "--dataset_root",
        type=str,
        default='/path/to/your/datasets/MiniImageNet/test',
        help="ImageFolder root directory",
    )
    parser.add_argument(
        "--blip_dir",
        type=str,
        default="/path/to/your/offline_models/blip-image-captioning-base",
        help="Local BLIP model directory",
    )
    parser.add_argument(
        "--clip_dir",
        type=str,
        default="/path/to/your/offline_models/clip-vit-base-patch32",
        help="Local CLIP model directory",
    )
    parser.add_argument(
        "--imagenet_index_json",
        type=str,
        default="/path/to/your/imagenet_class_index.json",
        help="Path to imagenet_class_index.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./semantic",
        help="Directory to save output files",
    )

    # ===== Naming & Config =====
    parser.add_argument('--dataset', type=str, default='MiniImageNet', 
                        choices=['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS'])
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="miniimagenet",
        help="Dataset name used for filenames (e.g., imagenet, cifar100)",
    )
    parser.add_argument(
        "--clsname_source",
        type=str,
        default="auto",
        choices=["auto", "imagenet_index", "folder_name"],
        help=(
            "How to get English class name:\n"
            "  - auto: MiniImageNet/TieredImageNet use imagenet_index.json，"
            "          CIFAR-FS/FC100 use folder name\n"
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="clip",
        choices=["clip"],
        help="Encoding mode",
    )
    parser.add_argument(
        "--text_type",
        type=str,
        default="caption",
        help="Text type for naming",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split: train / val / test",
    )

   
    parser.add_argument(
        "--level",
        type=str,
        default="both",
        choices=["both", "fine", "concat"],
        help="Control output files: both, fine, or concat",
    )

    # Semantics construction mode
    parser.add_argument(
        "--sem_mode",
        type=str,
        default="both",
        choices=["both", "fine", "concat"],
        help="Semantic construction mode",
    )


    # ===== Other Args =====
    parser.add_argument(
        "--max_images",
        type=int,
        default=-1,
        help="Process only first N images if > 0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    


    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # === Step 0: Configure Filenames ===
    print(">>>>>>>>>>>>>>>>>>>>>Step 0: Configuring filenames...")

    # 1. Class Name Semantic
    class_name_sem_filename = f"{args.dataset_name}_semantic_clsname_{args.mode}_{args.text_type}_{args.split}.pth"
    class_name_sem_path = os.path.join(args.output_dir, class_name_sem_filename)

    # 2. Instance-level Caption Mean Semantic
    class_caption_sem_filename = f"{args.dataset_name}_semantic_captionavg_{args.mode}_{args.text_type}_{args.split}.pth"
    class_caption_sem_path = os.path.join(args.output_dir, class_caption_sem_filename)

    # 3. Concat Semantic (Class + Caption)
    class_sem_filename = f"{args.dataset_name}_semantic_{args.mode}_{args.text_type}_{args.split}.pth"
    class_sem_path = os.path.join(args.output_dir, class_sem_filename)

    # 4. Image-level Semantic (Query)
    img_sem_filename = f"{args.dataset_name}_imgsem_{args.mode}_{args.text_type}_{args.split}.pth"
    img_sem_path = os.path.join(args.output_dir, img_sem_filename)

    

    print(f"[Info] class-name semantic (1)   pth: {class_name_sem_path}")
    print(f"[Info] caption-avg semantic (2) pth: {class_caption_sem_path}")
    print(f"[Info] concat semantic (3)      pth: {class_sem_path}")
    print(f"[Info] image-level semantic (4) pth: {img_sem_path}")
    print(f"[Info] level = {args.level}")


    # === Step 1: Load Data & Mappings ===
    print(">>>>>>>>>>>>>>>>>>>>>Step 1: Loading class mappings and dataset...")
    
    # 1) Load Dataset
    dataset = ImageFolder(args.dataset_root)
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"Found {len(class_to_idx)} classes, {len(dataset)} images.")

    num_images = len(dataset) if args.max_images <= 0 else min(len(dataset), args.max_images)
    print(f"Will process {num_images} / {len(dataset)} images...")

    # 2) Construct wnid_to_name based on dataset + clsname_source
    dataset_lower = args.dataset.lower()

    use_imagenet_index = (
        args.clsname_source == "imagenet_index"
        or (args.clsname_source == "auto" and dataset_lower in ["miniimagenet", "tieredimagenet"])
    )
    use_folder_name = (
        args.clsname_source == "folder_name"
        or (args.clsname_source == "auto" and dataset_lower in ["cifar-fs", "fc100"])
    )

    if use_imagenet_index:
        wnid_to_name = load_wnid_to_name(args.imagenet_index_json)
        print(f"[Info] Loaded {len(wnid_to_name)} wnid-name pairs from imagenet_class_index.json")
    elif use_folder_name:
        wnid_to_name = {
            class_name: class_name.replace("_", " ")
            for class_name in class_to_idx.keys()
        }
        print(f"[Info] Built {len(wnid_to_name)} class-name pairs from folder names")
    else:
        wnid_to_name = {}
        print("[Warn] wnid_to_name is empty: no explicit class-name prompts will be used.")


    # === Step 2: Load Models ===
    print(">>>>>>>>>>>>>>>>>>>>>Step 2: Loading BLIP and CLIP models...")
    
    print(f"Loading BLIP from: {args.blip_dir}")
    blip_processor, blip_model = load_blip(args.blip_dir, device)

    print(f"Loading CLIP from: {args.clip_dir}")
    clip_tokenizer, clip_model = load_clip(args.clip_dir, device)

    # === Step 3: Prepare Containers ===
    print(">>>>>>>>>>>>>>>>>>>>>Step 3: Preparing data containers...")
    
    

    # Class-level containers
    class2_caption_embeds = defaultdict(list)   # For 2: caption-only mean
    class2_concat_embeds = defaultdict(list)    # For 3: class+caption mean

    # Image-level container
    img_semantics = torch.zeros(num_images, 512, dtype=torch.float32)
    img_index_to_path = []
    img_index_to_class_idx = []


    # === Step 4: Main Loop (Generate & Encode) ===
    print(">>>>>>>>>>>>>>>>>>>>>Step 4: Processing images (Generating captions & Encoding)...")
    
    for idx in tqdm(range(num_images)):
        img_path, class_idx = dataset.imgs[idx]
        class_name = idx_to_class[class_idx]
        rel_path = os.path.relpath(img_path, args.dataset_root)

        image = Image.open(img_path).convert("RGB")

        # 4.1 BLIP Generate Caption
        caption = generate_caption(
            image=image,
            processor=blip_processor,
            blip_model=blip_model,
            device=device,
            max_new_tokens=30,
        )
        caption_clean = " ".join(caption.split())

        
        
        if class_name in wnid_to_name:
            english_name = wnid_to_name[class_name]
            caption_with_class = f"A photo of a {english_name}.{caption_clean}."  
        else:
            english_name = None
            caption_with_class = caption_clean

        

        # 4.2 Image-level Semantic (4)
        text_for_clip_img = caption_clean
        emb_img = encode_text_clip(
            [text_for_clip_img],
            tokenizer=clip_tokenizer,
            clip_model=clip_model,
            device=device,
        )[0]  # (512,)
        
        img_semantics[idx] = emb_img
        img_index_to_path.append(rel_path)
        img_index_to_class_idx.append(int(class_idx))

        # Accumulate for class-level caption mean (2)
        class2_caption_embeds[class_name].append(emb_img)

        # 4.3 Class-level Semantic (3)
        if class_name in wnid_to_name:
            english_name = wnid_to_name[class_name]
            text_for_clip_cls = f"A photo of a {english_name}.{caption_clean}"
        else:
            text_for_clip_cls = caption_clean

        emb_cls = encode_text_clip(
            [text_for_clip_cls],
            tokenizer=clip_tokenizer,
            clip_model=clip_model,
            device=device,
        )[0]
        class2_concat_embeds[class_name].append(emb_cls)



    # === Step 5: Construct Class-level Semantics ===
    print(">>>>>>>>>>>>>>>>>>>>>Step 5: Aggregating class-level semantics...")

    # 5.1 Class Name Embedding (1)
    semantic_class = {}
    for class_name in class_to_idx.keys():
        if class_name in wnid_to_name:
            english_name = wnid_to_name[class_name]
            text_for_class = f"A photo of a {english_name}."
        else:
            text_for_class = class_name

        emb_class = encode_text_clip(
            [text_for_class],
            tokenizer=clip_tokenizer,
            clip_model=clip_model,
            device=device,
        )[0]
        semantic_class[class_name] = emb_class

    # 5.2 Caption-only Mean (2)
    semantic_caption_avg = {}
    for class_name, emb_list in class2_caption_embeds.items():
        if len(emb_list) == 0:
            continue
        stacked = torch.stack(emb_list, dim=0)   # (N_c, 512)
        mean_emb = stacked.mean(dim=0)           # (512,)
        semantic_caption_avg[class_name] = mean_emb

    # 5.3 Class+Caption Mean (3)
    semantic_concat = {}
    for class_name, emb_list in class2_concat_embeds.items():
        if len(emb_list) == 0:
            continue
        stacked = torch.stack(emb_list, dim=0)   # (N_c, 512)
        mean_emb = stacked.mean(dim=0)           # (512,)
        semantic_concat[class_name] = mean_emb

    # === Step 6: Save Results ===
    print(">>>>>>>>>>>>>>>>>>>>>Step 6: Saving semantic files...")

    # Save 1: Class name embedding
    if args.level in ("both", "fine"):
        class_name_semantic_dict = {
            "semantic_class": semantic_class,     # 1: class name/def embedding
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
        }
        torch.save(class_name_semantic_dict, class_name_sem_path)
        print(f"[Save] class-name semantic (1) to: {class_name_sem_path}")

    # Save 2: Instance-level caption mean
    if args.level in ("both", "fine"):
        class_caption_semantic_dict = {
            "semantic_caption_avg": semantic_caption_avg,  # 2: caption mean
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
        }
        torch.save(class_caption_semantic_dict, class_caption_sem_path)
        print(f"[Save] caption-avg semantic (2) to: {class_caption_sem_path}")

    # Save 3: Concat semantic
    if args.level in ("both", "concat"):
        class_concat_semantic_dict = {
            "semantic_feature": semantic_concat,   # Legacy key
            "semantic_concat": semantic_concat,    # Semantic key
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
        }
        torch.save(class_concat_semantic_dict, class_sem_path)
        print(f"[Save] concat semantic (3) to: {class_sem_path}")


    # Save 4: Image-level semantics (Always saved)
    img_semantic_dict = {
        "img_semantics": img_semantics,                 # (N, 512)
        "img_index_to_path": img_index_to_path,
        "img_index_to_class_idx": img_index_to_class_idx,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }
    torch.save(img_semantic_dict, img_sem_path)
    print(f"[Save] image-level semantic features (4) to: {img_sem_path}")


    

    print("Done.")


if __name__ == "__main__":
    main()