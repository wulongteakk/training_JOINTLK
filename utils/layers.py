
import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.utils import freeze_net

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
            lengths = mask_or_lengths.float().clamp(min=1)
        else:
            mask = mask_or_lengths.bool()
            lengths = (1 - mask.float()).sum(dim=1).clamp(min=1)
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1), 0.0)
        return masked_inputs.sum(dim=1) / lengths.unsqueeze(-1)


def dropout_mask(x: torch.Tensor, size: Tuple[int, ...], p: float) -> torch.Tensor:
    mask = x.new_full(size, 1 - p)
    mask.bernoulli_().div_(1 - p)
    return mask


class EmbeddingDropout(nn.Module):
    """
    Apply dropout with probabily `embed_p` to an embedding layer `emb`.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """

    def __init__(self, emb: nn.Embedding, embed_p: float):
        super().__init__()
        self.emb = emb
        self.embed_p = embed_p
        self.pad_idx = -1 if emb.padding_idx is None else emb.padding_idx

    def forward(self, words: torch.Tensor) -> torch.Tensor:
        if self.training and self.embed_p > 0:
            mask = dropout_mask(self.emb.weight, (self.emb.weight.size(0), 1), self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)


class RNNDropout(nn.Module):


    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = dropout_mask(x, (x.size(0), 1, x.size(2)), self.p)
        return x * mask


class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size=300, emb_size=300, hidden_size=300, num_layers=2, bidirectional=True,
                 emb_p=0.0, input_p=0.0, hidden_p=0.0, output_p=0.0, pretrained_emb=None, pooling=True, pad=False):
        super().__init__()
        if pad:
            raise NotImplementedError("Padding LSTMEncoder not supported")
        self.pooling = pooling

        self.emb = EmbeddingDropout(nn.Embedding(vocab_size, emb_size), emb_p)
        if pretrained_emb is not None:
            self.emb.emb.weight.data.copy_(pretrained_emb)
        else:
            bias = math.sqrt(6.0 / emb_size)
            nn.init.uniform_(self.emb.emb.weight, -bias, bias)
        self.input_dropout = nn.Dropout(input_p)
        self.output_dropout = nn.Dropout(output_p)
        hidden_per_direction = hidden_size // 2 if bidirectional else hidden_size
        self.rnn = nn.LSTM(emb_size, hidden_per_direction, num_layers=num_layers,
                           dropout=hidden_p, bidirectional=bidirectional, batch_first=True)
        self.max_pool = MaxPoolLayer()

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:

        embed = self.emb(inputs)
        embed = self.input_dropout(embed)
        packed = pack_padded_sequence(embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=inputs.size(1))
        outputs = self.output_dropout(outputs)
        return self.max_pool(outputs, lengths) if self.pooling else outputs




class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        attn = attn.masked_fill(mask, float('-inf'))
        return output, attn


class AttPoolLayer(nn.Module):

    def __init__(self, d_q: int, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.w_qs = nn.Linear(d_q, d_k)
        nn.init.normal_(self.w_qs.weight, mean=0.0, std=np.sqrt(2.0 / (d_q + d_k)))
        self.attention = MatrixVectorScaledDotProductAttention(np.power(d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        qs = self.w_qs(q)
        output, attn = self.attention(qs, k, k, mask=mask)
        return self.dropout(output), attn


class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head: int, d_q_original: int, d_k_original: int, dropout: float = 0.1):
        super().__init__()
        if d_k_original % n_head != 0:
            raise ValueError("d_k_original must be divisible by n_head")
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0.0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0.0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0.0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = q.size(0)
        len_k = k.size(1)

        qs = self.w_qs(q).view(bs, self.n_head, self.d_k)
        ks = self.w_ks(k).view(bs, len_k, self.n_head, self.d_k)
        vs = self.w_vs(k).view(bs, len_k, self.n_head, self.d_v)

        qs = qs.permute(1, 0, 2).contiguous().view(self.n_head * bs, self.d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(self.n_head * bs, len_k, self.d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(self.n_head * bs, len_k, self.d_v)

        if mask is not None:
            mask = mask.repeat(self.n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(self.n_head, bs, self.d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, self.n_head * self.d_v)
        return self.dropout(output), attn

class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class CustomizedEmbedding(nn.Module):
    def __init__(self, concept_num: int, concept_out_dim: int, concept_in_dim: int,
                 use_contextualized: bool = False, pretrained_concept_emb: Optional[torch.Tensor] = None,
                 freeze_ent_emb: bool = True, scale: float = 1.0, init_range: float = 0.02):
        super().__init__()
        self.scale = scale
        self.use_contextualized = use_contextualized
        self.concept_in_dim = concept_in_dim
        self.concept_out_dim = concept_out_dim
        if not use_contextualized:
            self.emb = nn.Embedding(concept_num, concept_in_dim)
            if pretrained_concept_emb is not None:
                self.emb.weight.data.copy_(pretrained_concept_emb)
            else:
                self.emb.weight.data.normal_(mean=0.0, std=init_range)
            if freeze_ent_emb:
                freeze_net(self.emb)
        else:
            self.emb = None

        if concept_in_dim != concept_out_dim:
            self.cpt_transform = nn.Linear(concept_in_dim, concept_out_dim)
            self.activation = GELU()
        else:
            self.cpt_transform = None
            self.activation = None

    def forward(self, index: torch.Tensor, contextualized_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_contextualized:
            if contextualized_emb is None:
                raise ValueError("contextualized_emb must be provided when use_contextualized=True")
            emb = contextualized_emb * self.scale
        else:
            emb = self.emb(index) * self.scale

        if self.cpt_transform is not None:
            emb = self.cpt_transform(emb)
            emb = self.activation(emb)
        return emb


class CQAttention(nn.Module):
    def __init__(self, block_hidden_dim: int):
        super().__init__()
        self.w4C = nn.Parameter(torch.empty(block_hidden_dim, 1))
        self.w4Q = nn.Parameter(torch.empty(block_hidden_dim, 1))
        self.w4mlu = nn.Parameter(torch.empty(1, 1, block_hidden_dim))
        nn.init.xavier_uniform_(self.w4C)
        nn.init.xavier_uniform_(self.w4Q)
        nn.init.xavier_uniform_(self.w4mlu)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, C: torch.Tensor, Q: torch.Tensor, Cmask: torch.Tensor, Qmask: torch.Tensor,
                return_att: bool = False):
        similarity = self._trilinear_for_attention(C, Q) + self.bias
        S1 = self._masked_softmax(similarity, Qmask.unsqueeze(1), dim=2)
        S2 = self._masked_softmax(similarity, Cmask.unsqueeze(-1), dim=1)

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
        subres0 = torch.matmul(C, self.w4C)
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2)
        subres0 = subres0.expand(-1, -1, Q.size(-2))
        subres1 = subres1.expand(-1, C.size(-2), -1)
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        return subres0 + subres1 + subres2

    def _masked_softmax(self, x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        mask = mask.float()
        x = x.masked_fill(mask == 0, float('-inf'))
        return torch.softmax(x, dim=dim)



class BiAttention(nn.Module):
    def __init__(self, block_hidden_dim: int):
        super().__init__()
        self.w4C = nn.Parameter(torch.empty(block_hidden_dim, 1))
        self.w4Q = nn.Parameter(torch.empty(block_hidden_dim, 1))
        self.w4mlu = nn.Parameter(torch.empty(1, 1, block_hidden_dim))
        nn.init.xavier_uniform_(self.w4C)
        nn.init.xavier_uniform_(self.w4Q)
        nn.init.xavier_uniform_(self.w4mlu)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, C: torch.Tensor, Q: torch.Tensor, Cmask: torch.Tensor, Qmask: torch.Tensor,
                return_att: bool = False):
        similarity = self._trilinear_for_attention(C, Q) + self.bias
        S1 = self._masked_softmax(similarity, Qmask.unsqueeze(1), dim=2)
        S2 = self._masked_softmax(similarity, Cmask.unsqueeze(-1), dim=1)

        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, C * A, C * B], dim=2)
        if return_att:
            return out, S2
        return out

    def _trilinear_for_attention(self, C: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        subres0 = torch.matmul(C, self.w4C)
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2)
        subres0 = subres0.expand(-1, -1, Q.size(-2))
        subres1 = subres1.expand(-1, C.size(-2), -1)
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        return subres0 + subres1 + subres2


    def _masked_softmax(self, x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        mask = mask.float()
        x = x.masked_fill(mask == 0, float('-inf'))
        return torch.softmax(x, dim=dim)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
