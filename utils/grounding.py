

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.utils import freeze_net

__all__ = [
    "MaxPoolLayer",
    "MeanPoolLayer",
    "dropout_mask",
    "EmbeddingDropout",
    "RNNDropout",
    "MatrixVectorScaledDotProductAttention",
    "MultiheadAttPoolLayer",
    "GELU",
    "CustomizedEmbedding",
    "CQAttention",
]

class MaxPoolLayer(nn.Module):
    """Max pooling over the sequence dimension."""

    def forward(self, inputs: torch.Tensor, mask_or_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = inputs.size()
        if mask_or_lengths.dim() == 1:
            mask = torch.arange(seq_len, device=inputs.device).unsqueeze(0)
            mask = mask.expand(batch_size, seq_len)
            mask = mask >= mask_or_lengths.unsqueeze(1)
        else:
            mask = mask_or_lengths.bool()
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1), float('-inf'))
        return masked_inputs.max(dim=1).values

class MeanPoolLayer(nn.Module):
    """Mean pooling over the sequence dimension."""

    def forward(self, inputs: torch.Tensor, mask_or_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = inputs.size()
        if mask_or_lengths.dim() == 1:
            mask = torch.arange(seq_len, device=inputs.device).unsqueeze(0)
            mask = mask.expand(batch_size, seq_len)
            mask = mask >= mask_or_lengths.unsqueeze(1)
            lengths = mask_or_lengths.float().clamp(min=1).float()
        else:
            mask = mask_or_lengths.bool()
            lengths = (1 - mask.float()).sum(dim=1).clamp(min=1)
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1), 0.0)
        return masked_inputs.sum(dim=1) / lengths.unsqueeze(-1)


def dropout_mask(x: torch.Tensor, size: Tuple[int, ...], p: float) -> torch.Tensor:
    if p <= 0:
        return x.new_ones(size)
    mask = x.new_empty(size).bernoulli_(1 - p)
    return mask.div_(1 - p)



class EmbeddingDropout(nn.Module):


    def __init__(self, emb: nn.Embedding, embed_p: float):
        super().__init__()
        self.emb = emb
        self.embed_p = embed_p
        self.pad_idx = -1 if emb.padding_idx is None else emb.padding_idx

    def forward(self, words: torch.Tensor) -> torch.Tensor:
        if self.training and self.embed_p > 0:
            mask = dropout_mask(self.emb.weight, (self.emb.weight.size(0), 1), self.embed_p)
            weight = self.emb.weight * mask
        else:
            weight = self.emb.weight
        return F.embedding(
            words,
            weight,
            self.pad_idx,
            self.emb.max_norm,
            self.emb.norm_type,
            self.emb.scale_grad_by_freq,
            self.emb.sparse,
        )


class RNNDropout(nn.Module):


    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        mask = dropout_mask(x, (x.size(0), 1, x.size(2)), self.p)
        return x * mask





class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = (q.unsqueeze(1) * k).sum(2) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, float("-inf"))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn



class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head: int, d_q: int, d_k: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_k % n_head == 0
        self.n_head = n_head

        self.d_k = d_k // n_head
        self.d_v = d_k // n_head

        self.w_qs = nn.Linear(d_q, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k, n_head * self.d_v)

        for w in (self.w_qs, self.w_ks, self.w_vs):
            nn.init.normal_(w.weight, mean=0, std=math.sqrt(2.0 / (w.weight.size(0) + w.weight.size(1))))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=math.sqrt(self.d_k))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, len_k, _ = k.size()
        q_proj = self.w_qs(q).view(batch_size, self.n_head, self.d_k)
        k_proj = self.w_ks(k).view(batch_size, len_k, self.n_head, self.d_k)
        v_proj = self.w_vs(k).view(batch_size, len_k, self.n_head, self.d_v)



        if mask is not None:
            mask = mask.repeat(self.n_head, 1)


        output, attn = self.attention(q_proj, k_proj, v_proj, mask=mask)
        output = output.view(self.n_head, batch_size, self.d_v).permute(1, 0, 2).contiguous()
        output = output.view(batch_size, self.n_head * self.d_v)

        output = self.dropout(output)
        return output, attn

class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))


class CustomizedEmbedding(nn.Module):
    def __init__(
        self,
        concept_num: int,
        concept_out_dim: int,
        use_contextualized: bool = False,
        concept_in_dim: Optional[int] = None,
        pretrained_concept_emb: Optional[torch.Tensor] = None,
        freeze_ent_emb: bool = True,
    ) -> None:
        super().__init__()
        self.use_contextualized = use_contextualized
        self.concept_emb = nn.Embedding(concept_num, concept_out_dim)

        if pretrained_concept_emb is not None:
            if isinstance(pretrained_concept_emb, np.ndarray):
                pretrained_concept_emb = torch.tensor(pretrained_concept_emb)
            self.concept_emb.weight.data.copy_(pretrained_concept_emb)
        else:
            nn.init.normal_(self.concept_emb.weight, mean=0.0, std=0.02)

        if freeze_ent_emb:
            freeze_net(self.concept_emb)

        if self.use_contextualized:
            if concept_in_dim is None:
                raise ValueError("concept_in_dim is required when use_contextualized=True")
            self.proj = nn.Linear(concept_in_dim, concept_out_dim)
        else:
            self.proj = None

    def forward(
            self, concept_ids: torch.Tensor, contextualized_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pad_mask = concept_ids < 0
        concept_ids = concept_ids.masked_fill(pad_mask, 0)
        static_emb = self.concept_emb(concept_ids)
        if pad_mask.any():
            static_emb = static_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        if not self.use_contextualized or contextualized_emb is None:
            return static_emb
        return static_emb + self.proj(contextualized_emb)




class CQAttention(nn.Module):
    def __init__(self, block_hidden_dim: int):
        super().__init__()
        self.w4C = nn.Parameter(torch.zeros(block_hidden_dim, 1))
        self.w4Q = nn.Parameter(torch.zeros(block_hidden_dim, 1))
        self.w4mlu = nn.Parameter(torch.zeros(1, 1, block_hidden_dim))
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w4C)
        nn.init.xavier_uniform_(self.w4Q)
        nn.init.xavier_uniform_(self.w4mlu)

    def forward(
            self,
            C: torch.Tensor,
            Q: torch.Tensor,
            Cmask: torch.Tensor,
            Qmask: torch.Tensor,
            return_att: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        similarity = self._trilinear_for_attention(C, Q) + self.bias
        S1 = self._masked_softmax(similarity, Qmask.unsqueeze(1))
        S2 = self._masked_softmax(similarity, Cmask.unsqueeze(-1))

        A = torch.bmm(S1, Q)
        ### A:  torch.Size([5, 200, 200])

        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        ### B:  torch.Size([5, 200, 200])

        out = torch.cat([C, A, C * A, C * B], dim=2)

        if return_att:
            return out, S2

        ### out:  torch.Size([5, 200, 800])
        return out

    def _trilinear_for_attention(self, C: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        subres0 = torch.matmul(C, self.w4C).expand(-1, -1, Q.size(1))
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand(-1, C.size(1), -1)
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        return subres0 + subres1 + subres2

    def _masked_softmax(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        x = x.masked_fill(mask == 0, float('-inf'))
        return torch.softmax(x, dim=-1)
