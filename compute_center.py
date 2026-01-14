
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# from data.tiered_imagenet import tieredImageNet
from model.res12 import Res12
from model.swin_transformer import swin_tiny
import torch.utils.data
from utils.utils import transform_val, transform_val_cifar, cluster
from utils.utils import transform_val_224_cifar, transform_val_224,val_cifar_trans_84

from utils.common_args import add_feature_preprocess_args  
from utils.common_utils import preprocess_features, compute_mean_var_dict,compute_mean_var  

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backbone == 'resnet':
        model = Res12(avg_pool=True, drop_block='ImageNet' in args.dataset).to(device)
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


    elif args.backbone == 'swin':
        model = swin_tiny().to(device)
        model_dict = model.state_dict()
        checkpoint = torch.load(args.model_path)['params']
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}

    print(len(checkpoint))
    model.load_state_dict(checkpoint)
    model.eval()

    data = {}
    batch_size = 128
    shuffle = True
    
    # train
    if args.dataset == 'MiniImageNet':
        trainset = ImageFolder('/path/to/your/datasets/MiniImageNet/train', transform=transform_val if args.backbone == 'resnet' else transform_val_224)
    elif args.dataset == 'FC100':
        trainset = ImageFolder('/path/to/your/datasets/FC100/train', transform=transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar)
    elif args.dataset == 'CIFAR-FS':
        trainset = ImageFolder('/path/to/your/cifar_fs/train', transform=val_cifar_trans_84 if args.backbone == 'resnet' else transform_val_224_cifar)
    elif args.dataset == 'TieredImageNet':
        trainset = tieredImageNet(setname='train', augment=False)

        if args.backbone == 'swin':
            trainset = ImageFolder('/path/to/your/datasets/tiered_imagenet/train', transform=transform_val_224)
    else:
        raise ValueError('Non-supported Dataset.')

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                              pin_memory=True)
    idx_to_class = trainset.class_to_idx
    idx_to_class = {k: v for v, k in idx_to_class.items()}
    for x, labels in tqdm(train_loader):
        labels = [idx_to_class[l.item()] for l in labels]
        with torch.no_grad():
            x = model(x.to(device))
            x = preprocess_features(x, args)
        for i, l in enumerate(labels):
            if l in data:
                data[l].append(x[i].detach().cpu().numpy())
            else:
                data[l] = [x[i].detach().cpu().numpy()]
                
    print('Finished train')

    
    center_mean = {}
    center_var = {}
    for k, v in data.items():
        
        feats = np.array(v, dtype=np.float32)  # (N_c, D)
        mu_c, var_c = compute_mean_var(feats, eps=1e-5)  
        center_mean[k] = mu_c
        center_var[k] = var_c

    if args.dataset == 'TieredImageNet':
        data = {k: v[:700] for k, v in data.items()}
        center_cluster = cluster(data, len(data), 700)
    else:
        center_cluster = cluster(data, len(data), 600)

   
    torch.save({
        'mean': center_mean,
        'var': center_var,          
        'cluster': center_cluster,
        'center': center_mean
    }, 'center_{}_{}.pth'.format(args.dataset, args.backbone))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS'])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--center', default='mean',
                        choices=['mean', 'cluster'])
    parser.add_argument('--backbone', default='swin',
                        choices=['resnet', 'swin'])
    
    add_feature_preprocess_args(parser)

    args = parser.parse_args()

    
    print(vars(args))
    if args.backbone == 'resnet':
        args.model_path = './checkpoints/ResNet-{}.pth'.format(args.dataset)
    elif args.backbone == 'swin':
        args.model_path = './checkpoints/Swin-Tiny-{}.pth'.format(args.dataset)
    main()
