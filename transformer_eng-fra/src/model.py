import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dropout: float = 0.1
    max_len: int = 128
    attention_type: str = "dot"  # dot | additive


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, attention_type: str = "dot") -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_type = attention_type

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if attention_type == "additive":
            self.add_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
            self.add_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
            self.add_v = nn.Parameter(torch.randn(self.head_dim) / math.sqrt(self.head_dim))
        elif attention_type != "dot":
            raise ValueError("attention_type must be 'dot' or 'additive'")

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _dot_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

    def _additive_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q_term = self.add_q(q).unsqueeze(-2)
        k_term = self.add_k(k).unsqueeze(-3)
        scores = torch.tanh(q_term + k_term)
        return torch.einsum("bhqkd,d->bhqk", scores, self.add_v)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self._reshape_heads(self.q_proj(query))
        k = self._reshape_heads(self.k_proj(key))
        v = self._reshape_heads(self.v_proj(value))

        if self.attention_type == "dot":
            scores = self._dot_scores(q, k)
        else:
            scores = self._additive_scores(q, k)

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        bsz, seq_len, _, _ = context.shape
        context = context.view(bsz, seq_len, self.d_model)
        out = self.out_proj(context)
        return out, attn_weights


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout, cfg.attention_type)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.d_model),
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.drop1(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout, cfg.attention_type)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout, cfg.attention_type)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.d_model),
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.drop2 = nn.Dropout(cfg.dropout)
        self.drop3 = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        cross_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.drop1(self_attn_out))
        cross_out, _ = self.cross_attn(x, memory, memory, cross_mask)
        x = self.norm2(x + self.drop2(cross_out))
        ff_out = self.ffn(x)
        x = self.norm3(x + self.drop3(ff_out))
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, src_pad_id: int, tgt_pad_id: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

        self.src_embed = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=src_pad_id)
        self.tgt_embed = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=tgt_pad_id)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len)
        self.dropout = nn.Dropout(cfg.dropout)

        self.encoder_layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_decoder_layers)])
        self.generator = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)

    def _src_mask(self, src_ids: torch.Tensor) -> torch.Tensor:
        # Shape: [B, 1, 1, S]
        return (src_ids != self.src_pad_id).unsqueeze(1).unsqueeze(2)

    def _tgt_mask(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        # Shape: [B, 1, T, T]
        bsz, tgt_len = tgt_ids.shape
        pad_mask = (tgt_ids != self.tgt_pad_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt_ids.device, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        return pad_mask & causal_mask

    def encode(self, src_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_mask = self._src_mask(src_ids)
        x = self.src_embed(src_ids) * math.sqrt(self.cfg.d_model)
        x = self.dropout(self.pos_enc(x))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x, src_mask

    def decode(self, tgt_ids: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        tgt_mask = self._tgt_mask(tgt_ids)
        # Shape: [B, 1, T, S]
        cross_mask = src_mask.expand(-1, -1, tgt_ids.size(1), -1)

        x = self.tgt_embed(tgt_ids) * math.sqrt(self.cfg.d_model)
        x = self.dropout(self.pos_enc(x))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, cross_mask)
        return x

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        memory, src_mask = self.encode(src_ids)
        dec_out = self.decode(tgt_ids, memory, src_mask)
        return self.generator(dec_out)

    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 50,
    ) -> torch.Tensor:
        self.eval()
        memory, src_mask = self.encode(src_ids)

        generated = torch.full((src_ids.size(0), 1), bos_id, dtype=torch.long, device=src_ids.device)
        finished = torch.zeros(src_ids.size(0), dtype=torch.bool, device=src_ids.device)

        for _ in range(max_new_tokens):
            dec_out = self.decode(generated, memory, src_mask)
            logits = self.generator(dec_out[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            finished |= next_token.squeeze(1).eq(eos_id)
            if finished.all():
                break

        return generated
