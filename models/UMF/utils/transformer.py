import torch
import torch.nn as nn
from .linear_attention import LinearMultiheadAttention


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, in_dim, hidden_size, dropout_rate=0.1):
        super(Embeddings, self).__init__()

        self.project = nn.Conv2d(in_dim, hidden_size, kernel_size=1, stride=1, padding=0)
        n_patches = 60 * 80
        

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B = x.shape[0]

        x = self.project(x)
        projected_shape = x.shape
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, projected_shape
    

class Embeddings3D(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, in_dim, hidden_size, dropout_rate=0.1):
        super(Embeddings3D, self).__init__()

        self.project = nn.Conv3d(in_dim, hidden_size, kernel_size=2, stride=2, padding=0)
        n_patches = 25 * 25 * 25
        

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B = x.shape[0]

        x = self.project(x)
        projected_shape = x.shape
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, projected_shape
    


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate=0.1):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,  hidden_size, seq_len, num_heads=8, mpl_drop=0.1, attn_drop=0.,  qkv_bias=False,):
        super(TransformerBlock, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.add_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, hidden_size*4, dropout_rate=mpl_drop)
        self.attn = LinearMultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True, seq_len=seq_len, dropout=attn_drop)

        # Linear layers for Q, K, V projections
        self.q_linear = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_linear = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_linear = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.apply(self._init_weights)

    def forward(self, x):
        h = x
        x_norm = self.attention_norm(x)

        # Project Q, K, V
        q = self.q_linear(x_norm)
        k = self.k_linear(x_norm)
        v = self.v_linear(x_norm)


        x, weights = self.attn(q, k, v, need_weights=True)

        x = x + h

        h = x
        x = self.ffn_norm(self.add_norm(x))
        x = self.ffn(x)
        x = x + h
        return x, weights
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    