import numpy as np
import torch
from torch.utils.data import DataLoader
from data.samplers import CategoriesSampler

from utils.utils import Cosine_classifier, count_95acc
from utils.common_utils import preprocess_features

def _get_class_names(labels, idx_to_class_map):
    
    return [idx_to_class_map[i.item()] for i in labels.cpu()]


def validate_shot(model, H, val_dataset, val_sem_input, args, device, shot, logger, writer, epoch):
    """
    Verify four strategies:
        1. Proto-only: Pure visual baseline (Proto_raw vs V_query)
        2. Phi-Sym: Symmetric enhancement (Proto_enhanced vs Z_query) (imgsem only)
        3. Phi-Asym: Asymmetric enhancement (Proto_raw vs Z_query) (imgsem only)
        4. Phi-Supp: Support-only enhancement, Query remains raw (Proto_enhanced vs V_query) (Applied to all semantic types)

    val_sem_input: Tensor (for imgsem) or Dict[str, Tensor] (for clsname/concat/captionavg)
    """
    is_imgsem = torch.is_tensor(val_sem_input)
    val_idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}

    if not is_imgsem and not isinstance(val_sem_input, dict):
         raise RuntimeError("val_sem_input must be either an image-level Tensor or a class-level Dict.")

    model.eval()
    H.eval()

    sampler = CategoriesSampler(
        val_dataset.targets,
        args.test_batch,
        args.test_way,
        shot + args.query
    )
    loader = DataLoader(
        dataset=val_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    label = torch.arange(args.test_way).repeat(args.query).to(device)

    all_acc_proto = []     
    all_acc_phi_sym = []   
    all_acc_phi_asym = []  
    all_acc_phi_supp = []  

    with torch.no_grad():
        for batch in loader:
            data, labels, indices = batch
            data = data.to(device)

            # 1.visual feature (V_query)
            feat = model(data)
            if feat.dim() > 2: feat = feat.view(feat.size(0), -1)
            feat = preprocess_features(feat, args)

            n_support = shot * args.test_way
            support = feat[:n_support] # V_support
            query = feat[n_support:]   # V_query (raw query feature)
            
            indices_support = indices[:n_support]
            labels_support = labels[:n_support]

            # 2. Extracting Support semantic features     s_support
            if is_imgsem:
                s_support = val_sem_input[indices_support].to(device).float()
            else:
                support_class_names = _get_class_names(labels_support, val_idx_to_class)
                s_support = torch.stack([
                    val_sem_input[name] for name in support_class_names
                ]).to(device).float()
                
            # 3. Support enhancement (Proto_enhanced)
            z_support,_ = H(s_support, support) 
            proto_enhanced = z_support.view(shot, args.test_way, -1).mean(dim=0) 

            # 4. Proto-only (V_raw vs V_query)
            proto_raw = support.view(shot, args.test_way, -1).mean(dim=0) 
            _, pred_proto = Cosine_classifier(proto_raw, query)
            all_acc_proto.append((pred_proto == label).float().mean().item())

            # 5.Phi-Supp (Support enhancement vs V_query)
            _, pred_phi_supp = Cosine_classifier(proto_enhanced, query)
            all_acc_phi_supp.append((pred_phi_supp == label).float().mean().item())


            # 6. Phi-Sym/Phi-Asym 
            if is_imgsem:
                
                s_query = val_sem_input[indices[n_support:]].to(device).float()
                z_query , _ = H(s_query, query) 
                
                # 6.1.  Phi-Sym (Proto_enhanced vs Z_query)
                _, pred_phi_sym = Cosine_classifier(proto_enhanced, z_query)
                all_acc_phi_sym.append((pred_phi_sym == label).float().mean().item())

                # 6.2.  Phi-Asym (Proto_raw vs Z_query)
                _, pred_phi_asym = Cosine_classifier(proto_raw, z_query)
                all_acc_phi_asym.append((pred_phi_asym == label).float().mean().item())


    #  results
    m_proto, ci_proto = count_95acc(np.array(all_acc_proto))
    m_supp, ci_supp = count_95acc(np.array(all_acc_phi_supp))
    
    if is_imgsem:
        m_sym, ci_sym = count_95acc(np.array(all_acc_phi_sym))
        m_asym, ci_asym = count_95acc(np.array(all_acc_phi_asym))
        
        logger.info(
            f"[Val-{shot}shot][ImgSem] Proto: {m_proto*100:.2f}%±{ci_proto * 100:.2f}%, "
            f"Sym: {m_sym*100:.2f}%±{ci_sym * 100:.2f}%, "
            f"Asym: {m_asym*100:.2f}%±{ci_asym * 100:.2f}%, "
            f"Supp: {m_supp*100:.2f}%±{ci_supp * 100:.2f}%" 
        )

        writer.add_scalars(f"val/{shot}shot", {
            "proto": m_proto, "sym": m_sym, "asym": m_asym, "supp": m_supp
        }, epoch)
        
    else:
        m_sym, ci_sym, m_asym, ci_asym = 0.0, 0.0, 0.0, 0.0 
        
        logger.info(
            f"[Val-{shot}shot][ClsSem] Proto: {m_proto*100:.2f}%±{ci_proto * 100:.2f}%, "
            f"Supp: {m_supp*100:.2f}%±{ci_supp * 100:.2f}%"
        )
        
        writer.add_scalars(f"val/{shot}shot", {
            "proto": m_proto, "supp": m_supp
        }, epoch)
 
    return m_proto, ci_proto, m_sym, ci_sym, m_asym, ci_asym, m_supp, ci_supp
