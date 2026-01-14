
import numpy as np
import torch
from utils.utils import Cosine_classifier, proto_classifier, LR

def apply_nonneg(x, shift=1e-6, use_relu=True):
    """
    Nonnegation processing of features
    x: torch.Tensor,  (B, D)
    
    """
    if use_relu:
        x = torch.relu(x)
    if shift is not None and shift != 0.0:
        x = x + shift
    
    x = torch.clamp(x, min=1e-12)
    return x

def power_transform_tukey(x, lam):
    
    if lam == 0.0:
        return torch.log(x)
    else:
        return torch.pow(x, lam)

def power_transform_boxcox(x, lam):
   
    if lam == 0.0:
        return torch.log(x)
    else:
        return (torch.pow(x, lam) - 1.0) / lam

def preprocess_features(x, args):
    
    # 1. non-negative processing
    x = apply_nonneg(x, shift=getattr(args, 'pt_shift', 1e-6), use_relu=True)

    pt_type = getattr(args, 'pt_type', 'tukey').lower()
    lam = float(getattr(args, 'pt_lam', 0.5))

    # 2. Power transformation
    if pt_type == 'none':
        return x
    elif pt_type == 'tukey':
        x = power_transform_tukey(x, lam)
    elif pt_type == 'boxcox':
        x = power_transform_boxcox(x, lam)
    else:
       
        return x

    return x

def compute_mean_var(features, eps=1e-6):
    """
    
    features: np.ndarray, (N_c, D)
    return (D,), var_diag (D,)
    """
    feats = np.asarray(features, dtype=np.float32)
    mu = feats.mean(axis=0)
    var_diag = feats.var(axis=0) + eps  
    return mu, var_diag

def compute_mean_var_dict(data_dict, eps=1e-6):
    
    mean_dict = {}
    var_dict = {}
    for k, v in data_dict.items():
        feats = np.array(v, dtype=np.float32)
        mu, var_diag = compute_mean_var(feats, eps=eps)
        mean_dict[k] = mu
        var_dict[k] = var_diag
    return mean_dict, var_dict


def evaluate_classifier(clf_type, proto, query, labels, device):
    """
    Unified interface for different classifiers.
    
    Args:
        clf_type (str): 'Cosine', 'EU', or 'LR'
        proto (Tensor): [Way, Dim] - Prototypes or Support Features
        query (Tensor): [Num_Query, Dim] - Query Features
        labels (Tensor): [Num_Query] - Ground Truth Labels (0..Way-1)
    """
    proto = proto.to(device)
    query = query.to(device)
    labels = labels.to(device)
    
    if clf_type == 'Cosine':
        # Cosine_classifier returns (logits, predict)
        _, pred = Cosine_classifier(proto, query)
        
    elif clf_type == 'EU':
        # proto_classifier (Euclidean) returns (dist, predict)
        _, pred = proto_classifier(proto, query)
        
    elif clf_type == 'LR':
        # LR in utils.py takes (support, support_y, query)
        # We treat 'proto' as 1-shot support samples for the LR 
        # (Since MAP/Enhancer outputs prototypes, not individual samples)
        way = proto.size(0)
        # Construct pseudo-labels for the prototypes: 0, 1, ..., Way-1
        proto_labels = torch.arange(way).to(device)
        
        # LR usually returns numpy array or tensor, handled inside utils.LR
        pred = LR(proto, proto_labels, query)
        pred = pred.to(device) # Ensure on device
        
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")

    # Calculate Accuracy
    acc = (pred == labels).float().mean().item()
    return acc