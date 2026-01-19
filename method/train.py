import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Add base directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.samplers import CategoriesSampler
# from data.tiered_imagenet import tieredImageNet 
from utils.logger import loggers
from model.res12 import Res12
from model.swin_transformer import swin_tiny
from utils.utils import transform_train, transform_val, transform_train_cifar, transform_val_cifar,transform_train_cifar_84
from utils.utils import transform_train_224, transform_val_224, transform_train_224_cifar, transform_val_224_cifar,val_cifar_trans_84

# Common args & utils
from utils.common_args import add_feature_preprocess_args, add_map_args
from utils.common_utils import preprocess_features
from utils.map_utils import (
    load_base_stats_for_map,
    load_semantic_for_split,
)

# Method imports
from method.QueryEnhancer_cross_attention import QueryEnhancer
from method.valdiate import validate_shot




def supervised_contrastive_loss(features, labels, temperature=0.1):
   
    device = features.device
    features = F.normalize(features, p=2, dim=1)
    
    logits = torch.div(torch.matmul(features, features.T), temperature)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    logits_mask = torch.scatter(
        torch.ones_like(mask), 1, 
        torch.arange(features.shape[0]).view(-1, 1).to(device), 0
    )
    mask = mask * logits_mask
    
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
    
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
    loss = -mean_log_prob_pos.mean()
    return loss






class IndexedImageFolder(ImageFolder):
    """
    Inherits from ImageFolder to return index in __getitem__.
    Used to retrieve semantic vectors via index later.
    """
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic Training Hyperparams 
    parser.add_argument('--max-epoch', type=int, default=50)
    parser.add_argument('--test-batch', type=int, default=600)

    # Episodic Validation Config
    parser.add_argument('--center', default='mean', choices=['mean'])
    parser.add_argument('--shot', type=int, default=1) # Default shot for logging, but we validate both
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=5)

    # Dimensions
    parser.add_argument('--feat-size', type=int, default=768)
    parser.add_argument('--semantic-size', type=int, default=512)

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step-size', type=int, default=30)

    # Semantic Config
    parser.add_argument('--mode', type=str, default='clip', choices=['clip'])
    parser.add_argument('--text_type', type=str, default='caption', choices=['caption'])
    parser.add_argument('--sem_kind', type=str, default='imgsem', choices=['imgsem', 'clsname', 'captionavg', 'concat'], 
                        help='Type of semantic feature for H: image-level(imgsem) or class-level(clsname/captionavg/concat)')

    # Dataset & Backbone
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS'])
    parser.add_argument('--backbone', type=str, default='swin',
                        choices=['resnet', 'swin'])

    # Add common args (Preprocessing & MAP)
    add_feature_preprocess_args(parser)
    add_map_args(parser)

    args = parser.parse_args()

    # --------- Path Setup ---------
    # NOTE: Ensure these paths exist on your machine!

    if args.backbone == 'resnet':
        args.model_path = './checkpoints/ResNet-{}.pth'.format(args.dataset)
    elif args.backbone == 'swin':
        args.model_path = './checkpoints/Swin-Tiny-{}.pth'.format(args.dataset)

    
    args.work_dir = '{}_{}_{}_{}_{}_{}'.format(
        args.backbone, args.dataset, args.mode, args.text_type, args.center, args.sem_kind
    )

    if args.dataset == 'TieredImageNet':
        args.num_workers = 0

    os.makedirs(args.work_dir, exist_ok=True)
    log = loggers(os.path.join(args.work_dir, 'train'))
    log.info("")
    log.info(vars(args))
    writer = SummaryWriter(os.path.join(args.work_dir, 'train_semantic'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Step 0: Data Preparation
    # =========================================================================
    
    log.info("")
    log.info("Step 0: Initializing Data Loaders and Datasets...")

    # Define transforms
    is_resnet = (args.backbone == 'resnet')
    
    # NOTE: Please modify the root paths below according to your local environment
    if args.dataset == 'MiniImageNet':
        args.val = '/path/to/your/MiniImageNet/val'
        args.train = '/path/to/your/MiniImageNet/train'
        
        tr_trans = transform_train if is_resnet else transform_train_224
        val_trans = transform_val if is_resnet else transform_val_224
        
        train_dataset = IndexedImageFolder(args.train, transform=tr_trans)
        val_dataset = IndexedImageFolder(args.val, transform=val_trans)

    elif args.dataset == 'FC100':
        args.val = '/path/to/your/FC100/val'
        args.train = '/path/to/your/FC100/train'
        
        tr_trans = transform_train_cifar if is_resnet else transform_train_224_cifar
        val_trans = transform_val_cifar if is_resnet else transform_val_224_cifar
        
        train_dataset = IndexedImageFolder(args.train, transform=tr_trans)
        val_dataset = IndexedImageFolder(args.val, transform=val_trans)

    elif args.dataset == 'CIFAR-FS':
        args.val = '/path/to/your/cifar_fs/val'
        args.train = '/path/to/your/cifar_fs/train'
        
        tr_trans = transform_train_cifar_84 if is_resnet else transform_train_224_cifar
        val_trans = val_cifar_trans_84 if is_resnet else transform_val_224_cifar
        
        train_dataset = IndexedImageFolder(args.train, transform=tr_trans)
        val_dataset = IndexedImageFolder(args.val, transform=val_trans)

    elif args.dataset == 'TieredImageNet':
        args.val = '/path/to/your/tiered_imagenet/val'
        args.train = '/path/to/your/tiered_imagenet/train'
        train_dataset = IndexedImageFolder(args.train, transform=transform_train_224)
        val_dataset = IndexedImageFolder(args.val, transform=transform_val_224)
    else:
        raise ValueError('Non-supported Dataset.')

    # Prepare Loaders
    val_idx_to_class = {k: v for v, k in val_dataset.class_to_idx.items()}
    idx_to_class = {k: v for v, k in train_dataset.class_to_idx.items()}

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, 
                              drop_last=False, num_workers=args.num_workers, pin_memory=True)
    
    val_sampler = CategoriesSampler(val_dataset.targets, args.test_batch, args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True)

    # =========================================================================
    # Step 1: Load Base Statistics (MAP Prior)
    # =========================================================================
    log.info("Step 1: Loading Base Statistics for MAP...")
    
    base_means, base_cov, base_class_keys, base_class_to_idx = load_base_stats_for_map(
        args, base_dir=BASE_DIR,log=log
    )

    # Construct proto_center dictionary (ClassName -> NumpyVector)
    proto_center = {
        cls_name: base_means[i] for i, cls_name in enumerate(base_class_keys)
    }
    log.info(f"[MAP] Base means shape: {base_means.shape}")

    # =========================================================================
    # Step 2: Load Semantic Features
    # =========================================================================
    log.info("Step 2: Loading Semantic Features (Train & Val)...")

    train_semantic_feat = None
    val_semantic_feat = None
    img_semantics_train = None
    img_semantics_val = None

    # 1. Load Train Split
    train_semantic_feat, train_c2i, train_i2c, _img_semantics_train = load_semantic_for_split(
        args, split="train", base_dir=BASE_DIR, map_location="cpu", sem_kind=args.sem_kind, log=log
    )
    
    # 2. Load Val Split
    val_semantic_feat, val_c2i, val_i2c, _img_semantics_val = load_semantic_for_split(
        args, split="val", base_dir=BASE_DIR, map_location="cpu", sem_kind=args.sem_kind, log=log
    )
    # Check 
    if train_semantic_feat is None:
        raise RuntimeError(f"[Semantic-Train] Semantic feature loading failed, sem_kind={args.sem_kind}")
    
    if args.sem_kind == 'imgsem':
    
        img_semantics_train = train_semantic_feat 
        img_semantics_val = val_semantic_feat
        
       
        if torch.is_tensor(img_semantics_train):
            img_semantics_train = img_semantics_train.float()
            
        if torch.is_tensor(img_semantics_val):
            img_semantics_val = img_semantics_val.float()
    else:
        train_semantic_feat = {k: v.float() for k, v in train_semantic_feat.items()}
        val_semantic_feat = {k: v.float() for k, v in val_semantic_feat.items()}
        
    # Check alignment
    if args.sem_kind != 'imgsem':
        train_sem_classes = set(train_c2i.keys()) if 'c2i' in locals() and train_c2i else set(train_semantic_feat.keys())
        train_data_classes = set(train_dataset.class_to_idx.keys())
        if train_sem_classes and (train_sem_classes != train_data_classes):
            raise ValueError("[Semantic-Train] Class mismatch between semantics and dataset.")


    # Validation: Ensure base priors cover all training classes
    train_class_names = set(idx_to_class.values())
    missing_in_proto = train_class_names - set(proto_center.keys())
    if missing_in_proto:
        raise ValueError(f"[MAP] Missing base classes in center file: {missing_in_proto}")

    # =========================================================================
    # Step 3: Initialize Models (Backbone & Enhancer)
    # =========================================================================
    log.info("Step 3: Initializing Backbone and QueryEnhancer...")

    # Load Backbone
    if args.backbone == 'resnet':
        model = Res12(avg_pool=True, drop_block='ImageNet' in args.dataset).to(device)
    elif args.backbone == 'swin':
        model = swin_tiny().to(device)

    # Load Backbone Checkpoint
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Backbone checkpoint not found at: {args.model_path}")
        
    model_dict = model.state_dict()
    checkpoint = torch.load(args.model_path)['params']
    raw_ckpt = torch.load(args.model_path, map_location=device)
    if 'params' in raw_ckpt:
        state_dict = raw_ckpt['params']
    elif 'state_dict' in raw_ckpt:
        state_dict = raw_ckpt['state_dict']
    else:
        state_dict = raw_ckpt

    model_dict = model.state_dict()
    new_state_dict = {}
    print("\n")
    print(f"[Debug] Start loading weights from {args.model_path}")
    
    for k, v in state_dict.items():
        original_k = k
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('encoder.'):
            k = k[8:]
            
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                new_state_dict[k] = v
            else:
                print(f"  [Skip] Shape mismatch for {k}: ckpt {v.shape} vs model {model_dict[k].shape}")
        else:
            # print(f"  [Skip] Unused key: {original_k}") # Comments can be uncommented during debugging.
            pass

    # 4. Load weights (Key: strict=False, allows ignoring the classification header)
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"[Success] Model loaded. Missing keys (ignore FC/classifier): {len(msg.missing_keys)}")

    model.eval()
    log.info(f"Backbone loaded from {args.model_path}")

    # Initialize Enhancer (H) & Aux Classifier
    # Explicit dimension check
    if args.backbone == 'resnet': 
        feat_size = 640
    elif args.backbone == 'swin': 
        feat_size = 768
    else:
        feat_size = args.feat_size # fallback

    log.info(f"Feature Dimension set to: {feat_size}")

    H = QueryEnhancer(feat_size, args.semantic_size, hidden_dim=2048, drop=args.drop).to(device)
    num_base_classes = len(idx_to_class)
    aux_classifier = torch.nn.Linear(feat_size, num_base_classes).to(device)

    # Optimizer Setup
    optimizer = torch.optim.Adam(
        list(H.parameters()) + list(aux_classifier.parameters()), 
        lr=args.lr
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    # =========================================================================
    # Step 4: Training Loop
    # =========================================================================
    log.info("Step 4: Starting Training...")

    max_acc_1shot = 0.0
    max_acc_5shot = 0.0

    for epoch in range(args.max_epoch):
        H.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_con_loss = 0.0
        epoch_cls_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            if len(batch) != 3:
                raise RuntimeError("Dataset must return (img, label, index). Use IndexedImageFolder.")
            data, labels, indices = batch
            data = data.to(device)
            labels = labels.to(device)

            # 1. Extract Visual Features
            with torch.no_grad():
                feat = model(data)
                if feat.dim() > 2:
                    feat = feat.view(feat.size(0), -1)
                feat = preprocess_features(feat, args) # Non-negative + Power Transform
            v = feat

            # 2. Extract Semantic Features
            if args.sem_kind == 'imgsem':
                if img_semantics_train is None:
                    raise RuntimeError("img_semantics_train is None. Check semantic loading.")
                
                if torch.is_tensor(img_semantics_train):
                    s = img_semantics_train[indices].to(device)
                else:
                    # Fallback for list/dict-like indexable
                    s = torch.stack([img_semantics_train[int(i)] for i in indices]).to(device)
            else:
                class_names = [idx_to_class[l.item()] for l in labels.cpu()]
                s = torch.stack([
                    train_semantic_feat[name] for name in class_names
                ]).to(device)

            

            # 3. Target Prototype (Global Mean)
            proto_np = np.array([proto_center[idx_to_class[l.item()]] for l in labels.cpu()])
            proto = torch.from_numpy(proto_np).float().to(device)

            # 4. Forward & Loss
            optimizer.zero_grad()
            z, s_proj = H(s, v)

            
            con_loss = supervised_contrastive_loss(s_proj, labels, temperature=0.1) 
            lambda_con = 1.0           
            
            recon_loss = F.l1_loss(z, proto)
            
            logits = aux_classifier(z)
            cls_loss = ce_loss_fn(logits, labels)

            total_loss = recon_loss + 1.0 * cls_loss + lambda_con * con_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_con_loss += con_loss.item()
            epoch_cls_loss += cls_loss.item()
            num_batches += 1

        # Logging
        mean_loss = epoch_loss / max(num_batches, 1)
        mean_recon = epoch_recon_loss / max(num_batches, 1)
        mean_con = epoch_con_loss / max(num_batches, 1)
        mean_cls = epoch_cls_loss / max(num_batches, 1)
        current_alpha = H.alpha.item()

        log.info(
            f"[Epoch {epoch}/{args.max_epoch}] Total: {mean_loss:.4f} | "         
            f"L1: {mean_recon:.4f} | CE: {mean_cls:.4f} | Con: {mean_con:.4f} | Alpha: {current_alpha:.4f}"
        )
        writer.add_scalar("train/total_loss", mean_loss, epoch)
        writer.add_scalar("train/con_loss", mean_con, epoch)
        writer.add_scalar("train/alpha", current_alpha, epoch)
        lr_scheduler.step()

        # =========================================================================
        # Validation & Checkpointing
        # =========================================================================
        if (epoch % 1 == 0) or (epoch == args.max_epoch - 1):
            log.info(f"==> Validating at epoch {epoch}...")

            val_sem_input = img_semantics_val if args.sem_kind == 'imgsem' else val_semantic_feat
            
            # Validate 1-shot (Use Sym Acc as metric)
            m_proto_1, ci_proto_1, m_sym_1, ci_sym_1, m_asym_1, ci_asym_1, m_supp_1, ci_supp_1 = validate_shot( 
                model, H, val_dataset, val_sem_input, args, device, 1, log, writer, epoch
            )
            
            # Validate 5-shot (Use Sym Acc as metric)
            m_proto_5, ci_proto_5, m_sym_5, ci_sym_5, m_asym_5, ci_asym_5, m_supp_5, ci_supp_5 = validate_shot( 
                model, H, val_dataset, val_sem_input, args, device, 5, log, writer, epoch
            )
            
            
            if args.sem_kind == 'clsname':
                # Save Best 1-shot Model
                if m_supp_1 > max_acc_1shot:
                    max_acc_1shot = m_supp_1
                    save_path_1 = os.path.join(args.work_dir, f'epoch_best_1shot.pth')
                    torch.save({
                        'epoch': epoch,
                        'H_state_dict': H.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'aux_classifier': aux_classifier.state_dict(),
                        'acc_1shot': max_acc_1shot,
                    }, save_path_1)
                    log.info(f"==> New Best 1-shot Model Saved! Acc: {max_acc_1shot*100:.2f}% (Sem Kind: {args.sem_kind} -> Monitoring Supp Acc)")

                # Save Best 5-shot Model
                if m_supp_5 > max_acc_5shot:
                    max_acc_5shot = m_supp_5
                    save_path_5 = os.path.join(args.work_dir, f'epoch_best_5shot.pth')
                    torch.save({
                        'epoch': epoch,
                        'H_state_dict': H.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'aux_classifier': aux_classifier.state_dict(),
                        'acc_5shot': max_acc_5shot,
                    }, save_path_5)
                    log.info(f"==> New Best 5-shot Model Saved! Acc: {max_acc_5shot*100:.2f}% (Sem Kind: {args.sem_kind} -> Monitoring Supp Acc)")

            else:
                # Save Best 1-shot Model
                if m_sym_1 > max_acc_1shot:
                    max_acc_1shot = m_sym_1
                    save_path_1 = os.path.join(args.work_dir, f'epoch_best_1shot.pth') 
                    torch.save({
                        'epoch': epoch,
                        'H_state_dict': H.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'aux_classifier': aux_classifier.state_dict(),
                        'acc_1shot': max_acc_1shot,
                    }, save_path_1)
                    log.info(f"==> New Best 1-shot Model Saved! Acc: {max_acc_1shot*100:.2f}% (Sem Kind: {args.sem_kind})")

                # Save Best 5-shot Model
                if m_sym_5 > max_acc_5shot:
                    max_acc_5shot = m_sym_5
                    save_path_5 = os.path.join(args.work_dir, f'epoch_best_5shot.pth')
                    torch.save({
                        'epoch': epoch,
                        'H_state_dict': H.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'aux_classifier': aux_classifier.state_dict(),
                        'acc_5shot': max_acc_5shot,
                    }, save_path_5)
                    log.info(f"==> New Best 5-shot Model Saved! Acc: {max_acc_5shot*100:.2f}% (Sem Kind: {args.sem_kind})")
            # Save Last Model (Unchanged)
            if epoch == args.max_epoch - 1:
                save_path_last = os.path.join(args.work_dir, f'last_model.pth') # 增加 sem_kind
                torch.save({
                    'epoch': epoch,
                    'H_state_dict': H.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'aux_classifier': aux_classifier.state_dict(),
                    'acc_1shot': m_sym_1,
                    'acc_5shot': m_sym_5,
                }, save_path_last)
                log.info(f"==> Last Model Saved at Epoch {epoch} (Sem Kind: {args.sem_kind})")
            

    writer.close()
    log.info("Training Finished.")
