import torch
import torch.nn as nn

class QueryEnhancer(nn.Module):
    
    def __init__(self, feat_dim, sem_dim, hidden_dim=2048, drop=0.0):
        super().__init__()
        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.num_heads = 4
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=self.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feat_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, semantic, visual):
       
        v = visual.unsqueeze(1)  # [B, 1, D_v]       
        
        s_proj = self.sem_proj(semantic) # [B, D_v]
        s = s_proj.unsqueeze(1)   # [B, 1, D_v]
        
        # attn_out: [B, 1, D_v]
        attn_out, _ = self.attn(query=v, key=s, value=s)
        
        delta = attn_out.squeeze(1)
        delta = self.norm(delta) 
        
        z = visual + self.alpha * delta
        
        return z,s_proj 
