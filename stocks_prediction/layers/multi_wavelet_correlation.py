import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from stocks_prediction.layers.utils import get_filter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiWaveletTransform(nn.Module):
    """
    1D multiwavelet block.
    """

    def __init__(
        self,
        ich=1,
        k=8,
        alpha=16,
        c=128,
        nCZ=1,
        L=0,
        base="legendre",
        attention_dropout=0.1,
    ):
        super(MultiWaveletTransform, self).__init__()
        print("base", base)
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk0 = nn.Linear(ich, c * k)
        self.Lk1 = nn.Linear(c * k, ich)
        self.ich = ich
        self.MWT_CZ = nn.ModuleList(MWT_CZ1d(k, alpha, L, c, base) for i in range(nCZ))

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, : (L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        values = values.view(B, L, -1)

        V = self.Lk0(values).view(B, L, self.c, -1)
        for i in range(self.nCZ):
            V = self.MWT_CZ[i](V)
            if i < self.nCZ - 1:
                V = F.relu(V)

        V = self.Lk1(V.view(B, L, -1))
        V = V.view(B, L, -1, D)
        return (V.contiguous(), None)


class MultiWaveletCross(nn.Module):
    """
    1D Multiwavelet Cross Attention layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        seq_len_q,
        seq_len_kv,
        modes,
        c=64,
        k=8,
        ich=512,
        L=0,
        base="legendre",
        mode_select_method="random",
        initializer=None,
        activation="tanh",
        **kwargs,
    ):
        super(MultiWaveletCross, self).__init__()
        print("base", base)

        self.c = c
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        # Pure‐wavelet cross‐attention: ich must equal c * k (flattened dim = H*E)
        ich_dim = self.c * self.k
        self.attn1 = MultiWaveletTransform(
            ich=ich_dim,  # H*E
            k=self.k,  # per‐channel width
            c=self.c,  # number of channels
            L=self.L,
            base=base,
        )
        self.attn2 = MultiWaveletTransform(
            ich=ich_dim, k=self.k, c=self.c, L=self.L, base=base
        )
        self.attn3 = MultiWaveletTransform(
            ich=ich_dim, k=self.k, c=self.c, L=self.L, base=base
        )
        self.attn4 = MultiWaveletTransform(
            ich=ich_dim, k=self.k, c=self.c, L=self.L, base=base
        )
        self.T0 = nn.Linear(k, k)
        self.register_buffer("ec_s", torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer("ec_d", torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer("rc_e", torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer("rc_o", torch.Tensor(np.concatenate((H1r, G1r), axis=0)))

        self.Lk = nn.Linear(ich, c * k)
        self.Lq = nn.Linear(ich, c * k)
        self.Lv = nn.Linear(ich, c * k)
        self.out = nn.Linear(c * k, ich)
        self.modes1 = modes

    def forward(self, q, k, v, mask=None):
        B, N, H, E = q.shape  # (B, N, H, E) torch.Size([3, 768, 8, 2])
        _, S, _, _ = k.shape  # (B, S, H, E) torch.Size([3, 96, 8, 2])

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        q = self.Lq(q)
        q = q.view(q.shape[0], q.shape[1], self.c, self.k)
        k = self.Lk(k)
        k = k.view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.Lv(v)
        v = v.view(v.shape[0], v.shape[1], self.c, self.k)

        if N > S:
            zeros = torch.zeros_like(q[:, : (N - S), :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]

        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_q = q[:, 0 : nl - N, :, :]
        extra_k = k[:, 0 : nl - N, :, :]
        extra_v = v[:, 0 : nl - N, :, :]
        q = torch.cat([q, extra_q], 1)
        k = torch.cat([k, extra_k], 1)
        v = torch.cat([v, extra_v], 1)

        Ud_q = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_k = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_v = torch.jit.annotate(List[Tuple[Tensor]], [])

        Us_q = torch.jit.annotate(List[Tensor], [])
        Us_k = torch.jit.annotate(List[Tensor], [])
        Us_v = torch.jit.annotate(List[Tensor], [])

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        # decompose
        for i in range(ns - self.L):
            # print('q shape',q.shape)
            d, q = self.wavelet_transform(q)
            Ud_q += [tuple([d, q])]
            Us_q += [d]
        for i in range(ns - self.L):
            d, k = self.wavelet_transform(k)
            Ud_k += [tuple([d, k])]
            Us_k += [d]
        for i in range(ns - self.L):
            d, v = self.wavelet_transform(v)
            Ud_v += [tuple([d, v])]
            Us_v += [d]
        for i in range(ns - self.L):
            dk, sk = Ud_k[i], Us_k[i]
            dq, sq = Ud_q[i], Us_q[i]
            dv, sv = Ud_v[i], Us_v[i]
            Ud += [
                self.attn1(dq[0], dk[0], dv[0], mask)[0]
                + self.attn2(dq[1], dk[1], dv[1], mask)[0]
            ]
            Us += [self.attn3(sq, sk, sv, mask)[0]]
        v = self.attn4(q, k, v, mask)[0]

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)
            v = self.evenOdd(v)
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))
        return (v.contiguous(), None)

    def wavelet_transform(self, x):
        xa = torch.cat(
            [
                x[:, ::2, :, :],
                x[:, 1::2, :, :],
            ],
            -1,
        )
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


class FourierCrossAttentionW(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        seq_len_q,
        seq_len_kv,
        modes=16,
        activation="tanh",
        mode_select_method="random",
    ):
        super(FourierCrossAttentionW, self).__init__()
        print("corss fourier correlation used!")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.activation = activation

    def forward(self, q, k, v, mask):
        B, L, E, H = q.shape

        xq = q.permute(0, 3, 2, 1)  # size = [B, H, E, L] torch.Size([3, 8, 64, 512])
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)
        self.index_q = list(range(0, min(int(L // 2), self.modes1)))
        self.index_k_v = list(range(0, min(int(xv.shape[3] // 2), self.modes1)))

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(
            B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat
        )
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        xk_ft_ = torch.zeros(
            B, H, E, len(self.index_k_v), device=xq.device, dtype=torch.cfloat
        )
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_)
        if self.activation == "tanh":
            xqk_ft = xqk_ft.tanh()
        elif self.activation == "softmax":
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception(
                "{} actiation function is not implemented".format(self.activation)
            )
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)

        xqkvw = xqkv_ft
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        out = torch.fft.irfft(
            out_ft / self.in_channels / self.out_channels, n=xq.size(-1)
        ).permute(0, 3, 2, 1)
        # size = [B, L, H, E]
        return (out, None)


def naive_rfft(x: torch.Tensor) -> torch.Tensor:
    """
    Real-to-complex FFT implemented with an explicit DFT matrix.
    x: (..., N) real tensor
    returns: (..., K) complex tensor with K = N//2 + 1   (onesided)
    """
    *batch, N = x.shape
    K = N // 2 + 1
    n = torch.arange(N, device=x.device).reshape(1, N)  # (1, N)
    k = torch.arange(K, device=x.device).reshape(K, 1)  # (K, 1)
    # e^{-2π i n k / N}
    omega = torch.exp(-2j * math.pi * k @ n / N)  # (K, N)
    # (..., N) → (..., K)
    X = x.to(torch.cfloat) @ omega.T  # batch matmul
    return X  # (..., K) complex


def naive_irfft(X: torch.Tensor, N: int) -> torch.Tensor:
    """
    Inverse real FFT implemented with an explicit IDFT matrix.
    X: (..., K) complex (onesided) with K = N//2 + 1
    N: signal length to recover
    returns: (..., N) real tensor
    """
    *batch, K = X.shape
    assert K == N // 2 + 1, "Wrong spectrum length for irfft"

    # Rebuild the full length-N spectrum (Hermitian symmetry)
    X_full = torch.zeros(*batch, N, dtype=torch.cfloat, device=X.device)
    X_full[..., :K] = X
    if N % 2 == 0:  # even length: Nyquist term is unique
        X_full[..., K:] = torch.conj(torch.flip(X[..., 1:-1], dims=[-1]))
    else:  # odd length
        X_full[..., K:] = torch.conj(torch.flip(X[..., 1:], dims=[-1]))

    n = torch.arange(N, device=X.device).reshape(N, 1)  # (N, 1)
    k = torch.arange(N, device=X.device).reshape(1, N)  # (1, N)
    # e^{+2π i n k / N}
    omega = torch.exp(2j * math.pi * n @ k / N) / N  # (N, N)
    x_rec = (X_full @ omega).real  # (..., N)
    return x_rec


################################################################################
# Sparse Fourier kernel layer that uses the naïve FFTs above
################################################################################


class sparseKernelFT1d(nn.Module):
    """
    Identical API and numerics (to rounding) as the original layer,
    but **no torch.fft / torch.rfft / torch.irfft** are used.
    """

    def __init__(
        self, k: int, alpha: int, c: int = 1, nl: int = 1, initializer=None, **kwargs
    ):
        super().__init__()
        self.k = k
        self.modes1 = alpha
        self.scale = 1.0 / (c * k * c * k)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(c * k, c * k, alpha, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul1d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in_ch, modes) × (in_ch, out_ch, modes) → (batch, out_ch, modes)
        return torch.einsum("bim,iom->bom", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, N, c, k)  →  same shape on output
        """
        B, N, c, k = x.shape
        x = x.view(B, N, -1).permute(0, 2, 1)  # (B, c*k, N)

        # --- forward real FFT -------------------------------------------------
        x_fft = naive_rfft(x)  # (B, c*k, N//2+1)

        # keep only first l modes
        l_n = min(self.modes1, N // 2 + 1)
        out_ft = torch.zeros_like(x_fft)
        out_ft[..., :l_n] = self.compl_mul1d(
            x_fft[..., :l_n],  # (B, in_ch, l)
            self.weights1[..., :l_n],  # (in_ch, out_ch, l)
        )

        # --- inverse real FFT -------------------------------------------------
        x_rec = naive_irfft(out_ft, N)  # (B, c*k, N)

        # reshape back
        x_rec = x_rec.permute(0, 2, 1).view(B, N, c, k)
        return x_rec


# ##
class MWT_CZ1d(nn.Module):
    def __init__(
        self, k=3, alpha=64, L=0, c=1, base="legendre", initializer=None, **kwargs
    ):
        super(MWT_CZ1d, self).__init__()

        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.A = sparseKernelFT1d(k, alpha, c)
        self.B = sparseKernelFT1d(k, alpha, c)
        self.C = sparseKernelFT1d(k, alpha, c)

        self.T0 = nn.Linear(k, k)

        self.register_buffer("ec_s", torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer("ec_d", torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer("rc_e", torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer("rc_o", torch.Tensor(np.concatenate((H1r, G1r), axis=0)))

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, k)
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_x = x[:, 0 : nl - N, :, :]
        x = torch.cat([x, extra_x], 1)
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        #         decompose
        for i in range(ns - self.L):
            # print('x shape',x.shape)
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x)  # coarsest scale transform

        #        reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        x = x[:, :N, :, :]

        return x

    def wavelet_transform(self, x):
        xa = torch.cat(
            [
                x[:, ::2, :, :],
                x[:, 1::2, :, :],
            ],
            -1,
        )
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):

        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x
