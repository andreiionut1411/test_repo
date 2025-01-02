import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        positions = torch.arange(max_len).unsqueeze(1)

        scale = torch.arange(0, d_model, 2)
        scale = torch.exp(-torch.log(torch.tensor(10000.0)) * scale / d_model)
        pos_encodings = torch.zeros(max_len, d_model)

		# Even is sin, odd is cos
        pos_encodings[:, 0::2] = torch.sin(positions * scale)
        pos_encodings[:, 1::2] = torch.cos(positions * scale)
        pos_encodings = pos_encodings.unsqueeze(0)
        self.register_buffer('pos_encodings', pos_encodings)

    def forward(self, x):
        return x + self.pos_encodings[:, :x.size(1), :]


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()

        # We need epsilon for numerical stability in case std is 0, so we don't divide by 0
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Initalize Query, Key, Value, and Output parameters randomly
        self.W_q = nn.Parameter(torch.randn(d_model, d_model))
        self.W_k = nn.Parameter(torch.randn(d_model, d_model))
        self.W_v = nn.Parameter(torch.randn(d_model, d_model))
        self.W_o = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, x, mask=None):
        batch_size, seq_len, emb_dim = x.size()

        Q = torch.matmul(x, self.W_q)
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5

        # Apply autoregressive mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Apparently, when I tried without contiguous, it crashed, so now it's here
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        output = torch.matmul(attn_output, self.W_o)

        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, orig_dim, bigger_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(orig_dim, bigger_dim)
        self.linear2 = nn.Linear(bigger_dim, orig_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderBlock, self).__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

    def forward(self, x, mask):
        attention_out = self.attn(self.ln1(x), mask)
        x = x + attention_out
        feed_forward_out = self.ffn(self.ln2(x))
        x = x + feed_forward_out

        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=6, d_ff=8, num_layers=6, max_len=5000):
        super(DecoderOnlyTransformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.ln_final = LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def generate_autoregressive_mask(self, seq_len):
        return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        batch_size, sequence_length = x.size()
        mask = self.generate_autoregressive_mask(sequence_length).to(x.device)
        x = self.token_embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_final(x)
        logits = self.output_projection(x)
        return logits


def causal_language_model_loss(logits, targets, pad_token_idx):
    mask = (targets != pad_token_idx).float()
    vocab_size = logits.size(-1)
    logits = logits.view(-1, vocab_size)
    targets = targets.view(-1)
    mask = mask.view(-1)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fn(logits, targets)
    loss = loss * mask

    return loss.sum() / mask.sum()