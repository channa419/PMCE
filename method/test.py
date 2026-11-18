import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from torchvision.datasets import ImageFolder
from data.samplers import CategoriesSampler
from utils.logger import loggers
from model.res12 import Res12
from model.swin_transformer import swin_tiny
from utils.utils import set_seed, count_95acc, transform_val, transform_val_224, transform_val_cifar, transform_val_224_cifar, val_cifar_trans_84

# Common Utils
from utils.common_args import add_feature_preprocess_args, add_map_args
from utils.common_utils import preprocess_features, evaluate_classifier
from utils.map_utils import (
    MAP_estimation_semantic,
    load_base_stats_for_map,
    load_semantic_for_split,
)
from method.QueryEnhancer_cross_attention import QueryEnhancer

class IndexedImageFolder(ImageFolder):
    """
    Inherits from ImageFolder to return index in __getitem__.
    Allows retrieval of semantic vectors via index.
    """
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Testing Config
    parser.add_argument('--test-batch', type=int, default=600)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--center', default='mean', choices=['mean'])
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--num-workers', type=int, default=8)

    # Dimensions
    parser.add_argument('--feat-size', type=int, default=768)
    parser.add_argument('--semantic-size', type=int, default=512)

    # Semantic Config
    parser.add_argument('--mode', type=str, default='clip', choices=['clip'])
    parser.add_argument('--text_type', type=str, default='caption', choices=['caption'])
    
    # Dataset & Backbone
    parser.add_argument('--dataset', type=str, default='MiniImageNet', 
                        choices=['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS'])
    parser.add_argument('--sem_kind', type=str, default='imgsem', choices=['imgsem', 'clsname', 'captionavg', 'concat'])
    parser.add_argument('--backbone', type=str, default='swin', choices=['resnet', 'swin'])

    # Enhancer Config
    parser.add_argument('--use_enhancer', action='store_true')
    parser.add_argument('--enhancer_path', type=str, default='')

    # Add Common Args
    add_feature_preprocess_args(parser)
    add_map_args(parser)

    args = parser.parse_args()

    # 1. Check Dimensions
    if args.backbone == 'resnet': 
        args.feat_size = 640
    elif args.backbone == 'swin': 
        args.feat_size = 768
    
    # 2. Setup Paths
    if args.backbone == 'resnet':
        args.model_path = './checkpoints/ResNet-{}.pth'.format(args.dataset)
    elif args.backbone == 'swin':
        args.model_path = './checkpoints/Swin-Tiny-{}.pth'.format(args.dataset)

    if not args.enhancer_path:
        train_dir_name = '{}_{}_{}_{}_{}_{}'.format(
            args.backbone, args.dataset, args.mode, args.text_type, args.center, args.sem_kind
        )
        args.enhancer_path = os.path.join(train_dir_name, f'epoch_best_{args.shot}shot.pth')

    # Setup Logging
    log = loggers(f'test_{args.dataset}_{args.backbone}_{args.shot}shot')
    log.info(vars(args))
    set_seed(args.seed)

    if args.dataset == 'TieredImageNet':
        args.num_workers = 0

    # =========================================================================
    # Step 0: Data Loading
    # =========================================================================
    log.info("Step 0: Initializing Data Loaders...")
    
    if args.dataset == 'MiniImageNet':
        args.test = '/path/to/your/mini-imagenet-sxc/test'
        val_trans = transform_val if args.backbone == 'resnet' else transform_val_224
        test_dataset = IndexedImageFolder(args.test, transform=val_trans)
    elif args.dataset == 'FC100':
        args.test = '/path/to/your/FC100/test'
        val_trans = transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar
        test_dataset = IndexedImageFolder(args.test, transform=val_trans)
    elif args.dataset == 'CIFAR-FS':
        args.test = '/path/to/your/cifar_fs/test'
        val_trans = val_cifar_trans_84 if args.backbone == 'resnet' else transform_val_224_cifar
        test_dataset = IndexedImageFolder(args.test, transform=val_trans)
    elif args.dataset == 'TieredImageNet':
        args.test = '/path/to/your/tiered_imagenet/test'
        val_trans = transform_val_224
        test_dataset = IndexedImageFolder(args.test, transform=val_trans)
    else:
        raise ValueError('Non-supported Dataset.')

    idx_to_class = {k: v for v, k in test_dataset.class_to_idx.items()}
    val_sampler = CategoriesSampler(test_dataset.targets, args.test_batch, args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=test_dataset, batch_sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Step 1: Model Loading (Backbone & Caption Enhancer)
    # =========================================================================
    log.info("Step 1: Loading Models...")

    if args.backbone == 'resnet':
        model = Res12(avg_pool=True, drop_block='ImageNet' in args.dataset).to(device)
    elif args.backbone == 'swin':
        model = swin_tiny().to(device)
        
    raw_ckpt = torch.load(args.model_path, map_location=device)
    state_dict = raw_ckpt['params'] if 'params' in raw_ckpt else raw_ckpt.get('state_dict', raw_ckpt)

    model_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'): k = k[7:]
        if k.startswith('encoder.'): k = k[8:]
        if k in model_dict and v.shape == model_dict[k].shape:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    H_cap = None
    if args.use_enhancer:
        log.info(f"[Enhancer] Loading Caption Enhancer from {args.enhancer_path}")
        H_cap = QueryEnhancer(args.feat_size, args.semantic_size, hidden_dim=2048, drop=0.0).to(device)
        ckpt_cap = torch.load(args.enhancer_path, map_location=device)
        if 'H_state_dict' in ckpt_cap: H_cap.load_state_dict(ckpt_cap['H_state_dict'])
        elif 'G' in ckpt_cap: H_cap.load_state_dict(ckpt_cap['G'].state_dict())
        else: H_cap.load_state_dict(ckpt_cap)
        H_cap.eval()

    # =========================================================================
    # Step 2: Semantic Loading
    # =========================================================================
    log.info("Step 2: Loading Semantics...")

    base_means, base_cov, base_class_keys, _ = load_base_stats_for_map(args, base_dir=BASE_DIR, log=log)
    
    # Load Class Name semantics for MAP (Config A)
    base_sem_clsname, _, _, _ = load_semantic_for_split(args, split="train", base_dir=BASE_DIR, sem_kind="clsname", log=log)
    novel_sem_clsname_dict, _, _, _ = load_semantic_for_split(args, split="test", base_dir=BASE_DIR, sem_kind="clsname", log=log)
    
    # Load Image semantics for Enhancer
    img_semantics, _, _, _ = load_semantic_for_split(args, split="test", base_dir=BASE_DIR, sem_kind="imgsem", log=log)

    base_semantic_A = np.stack([base_sem_clsname[c].cpu().numpy() for c in base_class_keys], axis=0)
    novel_sem_clsname = {k: v.float().to(device) for k, v in novel_sem_clsname_dict.items()}
    if img_semantics is not None:
        img_semantics = img_semantics.float().to(device)

    # =========================================================================
    # Step 3: Episodic Testing 
    # =========================================================================
    log.info("Step 3: Starting Evaluation...")
    
    results = defaultdict(list)
    # Matches CategoriesSampler interleaved query order: [q0_c0..cN, q1_c0..cN, ...]
    label = torch.arange(args.test_way, device=device).repeat(args.query)

    with torch.no_grad():
        for batch in tqdm(val_loader):
            data, labels, indices = batch
            data = data.to(device)
            
            feats = model(data)
            feats = preprocess_features(feats, args).view(feats.size(0), -1)
            
            n_support = args.shot * args.test_way
            support, query = feats[:n_support], feats[n_support:]
            support_indices = indices[:n_support].to(device)
            query_indices = indices[n_support:].to(device)

            proto_visual = support.view(args.shot, args.test_way, -1).mean(dim=0)
            
            # CategoriesSampler layout after stack().t().reshape(-1):
            # [s0_c0, s0_c1, ..., s0_cN, s1_c0, ..., q0_c0, ...]
            # so class-t appears at indices t, t+way, t+2*way, ...
            # --- MAP Calculation (Name_Name) ---
            proto_list = []
            for t in range(args.test_way):
                global_idx = labels[t].item()
                class_name = idx_to_class[global_idx]
                
                curr_supp_sem = novel_sem_clsname[class_name].cpu().numpy()
                support_t = support.view(args.shot, args.test_way, -1)[:, t, :]
                
                mu_s, _ = MAP_estimation_semantic(
                    support_semantic=curr_supp_sem,
                    support_image=support_t.cpu().numpy(),
                    base_semantic=base_semantic_A,
                    base_means=base_means, base_cov=base_cov,
                    topk=args.map_topk, Estimate_COV=args.map_estimate_cov,
                    alpha=args.map_alpha, beta=args.map_beta, cov_bias=args.map_cov_bias,
                    tau=args.map_tau,
                    use_semantic=True
                )
                proto_list.append(torch.from_numpy(mu_s).float().to(device))
            map_proto = torch.stack(proto_list, dim=0)

            # --- Caption Enhancement ---
            strategies = {
                'Baseline': (proto_visual, query),
                'MAP': (map_proto, query),
            }

            if H_cap is not None:
                s_query_img = img_semantics[query_indices]
                z_query_cap, _ = H_cap(s_query_img, query)

                s_support_img = img_semantics[support_indices]
                s_proto_avg = s_support_img.view(args.shot, args.test_way, -1).mean(dim=0)

                z_proto_visual_cap, _ = H_cap(s_proto_avg, proto_visual)
                z_map_proto_cap, _ = H_cap(s_proto_avg.float().contiguous(), map_proto.float().contiguous())

                strategies.update({
                    'Phi_cap(S)':        (z_proto_visual_cap, query),
                    'Phi_cap(Q)':        (proto_visual, z_query_cap),
                    'Phi_cap(Both)':     (z_proto_visual_cap, z_query_cap),
                    'MAP+Phi_cap(S)':    (z_map_proto_cap, query),
                    'MAP+Phi_cap(Q)':    (map_proto, z_query_cap),
                    'MAP+Phi_cap(Both)': (z_map_proto_cap, z_query_cap),
                })

            for name, (p_s, p_q) in strategies.items():
                acc = evaluate_classifier('LR', p_s, p_q, label, device)
                results[name].append(acc)

    # =========================================================================
    # Step 4: Final Report
    # =========================================================================
    log.info("\n" + "="*80)
    log.info(" FINAL RESULTS (Classifier: LR | MAP: Name_Name | Enhancer: Caption)")
    log.info("="*80)
    log.info(f"{'Strategy':<25} | {'Accuracy':<20}")
    log.info("-" * 50)
    
    order = ['Baseline', 'MAP', 'Phi_cap(S)', 'Phi_cap(Q)', 'Phi_cap(Both)', 
             'MAP+Phi_cap(S)', 'MAP+Phi_cap(Q)', 'MAP+Phi_cap(Both)']
    
    for name in order:
        accs = results.get(name, [])
        if not accs:
            continue
        m, ci = count_95acc(np.array(accs))
        log.info(f"{name:<25} | {m*100:.2f} +- {ci*100:.2f}%")
    
    log.info("="*80)
    log.info("Done.")
