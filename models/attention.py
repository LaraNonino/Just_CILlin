import torch
from torch import nn
import torch.nn.functional as F

import math

class SelfAttention(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        q_dim=768,
        v_dim=768,
        collapse=False
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.q_dim = q_dim
        self.v_dim = v_dim
        
        self.W_q = nn.Linear(self.embed_dim, self.q_dim, bias=False)
        self.W_k = nn.Linear(self.embed_dim, self.q_dim, bias=False)
        self.W_v = nn.Linear(self.embed_dim, self.v_dim, bias=False)

        self.collapse = collapse

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_length, embed_dim) V (batch_size, embed_dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        H = self._scaled_dot_product_attention(Q, K, V) # H: (batch_size, embed_dim)
        
        if self.collapse:
            c = torch.sum(H, dim=1) # c: context vector
            return c
        return H
    
    def _scaled_dot_product_attention(self, Q, K, V):
        if len(Q.shape) == 2: # x:(batch_size, embed_dim)
            prod = torch.bmm(Q, K)
        else: # x: (batch_size, seq_length, embed_dim)
            prod = torch.bmm(Q, torch.transpose(K, -2, -1))

        scores = torch.div(
            prod,
            math.sqrt(self.embed_dim)
        ) # scale by embed_dim so that the softmax doesn't saturate

        A = F.softmax(scores, dim=-1)
        H = torch.bmm(A, V)

        return H