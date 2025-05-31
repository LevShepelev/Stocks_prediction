from math import sqrt

import numpy as np
import torch
import torch.nn as nn

from stocks_prediction.utils.masking import ProbMask, TriangularCausalMask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(
        self,
        Q: torch.Tensor,  # (B, H, L_Q, D)
        K: torch.Tensor,  # (B, H, L_K, D)
        sample_k: int,  # U_part  = c · ln(L_K)
        n_top: int,  # n_top   = c · ln(L_Q)
    ):
        """
        Probabilistic attention key-sampling (Informer).

        Returns
        -------
        Q_K  : Tensor  # (B, H, n_top, L_K)
        M_top: Tensor  # (B, H, n_top) indices of the kept queries
        """
        B, H, L_K, D = K.shape
        L_Q = Q.size(2)
        device = Q.device

        # ------------------------------------------------------------
        # 1) sample `sample_k` keys for every query position
        #    (rand ∈ [0,1) → indices < L_K)  – ONNX-friendly
        # ------------------------------------------------------------
        rand = torch.rand((L_Q, sample_k), device=device)
        index_sample = (rand * L_K).long()  # (L_Q, sample_k)

        # shape helpers for gather
        index_sample_exp = index_sample.view(
            1, 1, L_Q, sample_k, 1
        ).expand(  # add B,H,E dims
            B, H, -1, -1, D
        )  # (B,H,L_Q,sample_k,1)

        # gather sampled keys: (B, H, L_Q, sample_k, D)
        K_sample = torch.gather(
            K.unsqueeze(2).expand(-1, -1, L_Q, -1, -1),  # (B,H,L_Q,L_K,D)
            3,
            index_sample_exp,
        )

        # ------------------------------------------------------------
        # 2) Q · K_sample  →  (B, H, L_Q, sample_k)
        # ------------------------------------------------------------
        Q_K_sample = torch.matmul(  # (B,H,L_Q,1,D) x (B,H,L_Q,D,sample_k)
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)  # add key-slot dim
        ).squeeze(-2)

        # ------------------------------------------------------------
        # 3) sparsity measurement & pick top-n queries
        # ------------------------------------------------------------
        M = Q_K_sample.amax(-1) - Q_K_sample.mean(-1)  # (B,H,L_Q)
        M_top = M.topk(n_top, dim=-1, sorted=False).indices  # (B,H,n_top)

        # gather top queries: (B,H,n_top,D)
        Q_reduce = torch.gather(Q, 2, M_top.unsqueeze(-1).expand(-1, -1, -1, D))

        # ------------------------------------------------------------
        # 4) full dot-product for those queries only
        # ------------------------------------------------------------
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # (B,H,n_top,L_K)

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert L_Q == L_V  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        # ── debug line so you know this code is running ──
        print("[DEBUG] Using patched ProbAttention.forward")

        # your original beginning:
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        def _to_tensor(x, device):
            if torch.is_tensor(x):
                return x.to(device=device, dtype=torch.float32)
            return torch.tensor(float(x), device=device)

        L_K_t = _to_tensor(L_K, queries.device)
        L_Q_t = _to_tensor(L_Q, queries.device)

        U_part = int(self.factor * torch.ceil(torch.log(L_K_t)).item())
        n_top = int(self.factor * torch.ceil(torch.log(L_Q_t)).item())
        U_part = min(U_part, L_K)
        n_top = min(n_top, L_Q)

        scores_top, index = self._prob_QK(
            queries,
            keys,
            sample_k=U_part,
            n_top=n_top,
        )  # (B, H, n_top, L_K)

        # ── HERE: catch *any* small-first dimension Add that ONNX will choke on ──
        # reshape into (B*H, n_top, L_K)
        st = scores_top.reshape(B * H, n_top, L_K)

        # find *all* attributes of shape (H, L_K) or (H,1) and expand them:
        for name, tensor in self.__dict__.items():
            if (
                isinstance(tensor, torch.Tensor)
                and tensor.ndim == 2
                and tensor.shape[0] == H
            ):
                # assume this is a bias or relative-position thing
                bias = tensor  # shape (H, L_K) or (H,1)
                # expand to (B*H, 1, L_K)
                b = bias.unsqueeze(0).repeat(B, 1, 1).reshape(B * H, 1, tensor.size(1))
                # safe add
                st = st + b

        # put back into (B, H, n_top, L_K)
        scores_top = st.view(B, H, n_top, L_K)

        # your original scale + context update:
        scale = self.scale or 1.0 / sqrt(D)
        scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask
        )

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
