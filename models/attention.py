import torch
from torch import nn
import torch.nn.functional as F

import math

class SelfAttention3D(nn.Module):
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
        # x: (batch_size, seq_length, embed_dim)
        Q = self.W_q(x) # Q: (batch_size, seq_len, q_dim)
        K = self.W_k(x) # K: (batch_size, seq_len, k_dim=q_dim)
        V = self.W_v(x) # V: (batch_size, seq_len, v_dim)
        
        H = self._scaled_dot_product_attention(Q, K, V) 
        
        if self.collapse:
            c = torch.sum(H, dim=1) # c: context vector
            return c
        return H
    
    def _scaled_dot_product_attention(self, Q, K, V):
        scores = torch.div(
            torch.matmul(Q, torch.transpose(K, -2, -1)), 
            math.sqrt(self.q_dim)
        ) # scale by embed_dim so that the softmax doesn't saturate

        A = F.softmax(scores, dim=-1) # A: (batch_size, seq_len, seq_len)
        H = torch.bmm(A, V) #  H: (batch_size, seq_len, v_dim)
        
        return H
    
class SelfAttention2D(nn.Module):
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
        # x: (batch_size, embed_dim)
        Q = self.W_q(x) # Q: (batch_size, q_dim)
        K = self.W_k(x) # K: (batch_size, k_dim=q_dim)
        V = self.W_v(x) # V: (batch_size, v_dim)
        
        H = self._scaled_dot_product_attention(Q, K, V) # H: (batch_size, embed_dim)
        
        if self.collapse:
            c = torch.sum(H, dim=1) # c: context vector
            return c
        return H
    
    def _scaled_dot_product_attention(self, Q, K, V):
        # x: (batch_size, seq_length, embed_dim)
        scores = torch.div(
            torch.matmul(Q, torch.transpose(K), -2, -1), 
            math.sqrt(self.q_dim)
        ) # scale by embed_dim so that the softmax doesn't saturate

        A = F.softmax(scores, dim=-1) # A: (batch_size, seq_len, seq_len)
        H = torch.bmm(A, V) #  H: (batch_size, seq_len, v_dim)
        
        return H
    
# x = torch.ones((10, 7, 16))
# att = SelfAttention3D(16, q_dim=30, v_dim=8)
# x = att(x)
# print(x.shape)

# x = torch.ones((10, 16))
# att = SelfAttention2D(16, q_dim=30, v_dim=8)
# x = att(x)
# print(x.shape)