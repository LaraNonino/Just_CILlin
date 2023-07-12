import torch
from torch import nn
import torch.nn.functional as F

import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__(embed_dim)

        self.embed_dim = embed_dim
        
        self.W_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.W_k = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.W_v = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(self, x: torch.Tensor):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        H = self._scaled_dot_product_attention(Q, K, V) # H: (batch_size, embed_dim)
        # c = torch.sum(H, dim=1) # c: context vector

        return H
    
    def _scaled_dot_product_attention(self, Q, K, V):

        # scale by embed_dim so that the softmax doesn't saturate
        Q = torch.div(Q, math.sqrt(self.embed_dim))
        scores = torch.bmm(Q, torch.transpose(K, -2, -1))

        A = F.softmax(scores, dim=-1)
        H = torch.bmm(A, V)

        return H