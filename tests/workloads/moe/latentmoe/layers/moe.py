"""Mixture-of-Experts layer: DeepSeek-V4 routing x Kimi-K3 Stable LatentMoE.

One module covers both families through config:

  * fine-grained routed experts + always-on shared experts  (DeepSeekMoE lineage)
  * router scores:  Sigmoid(W_r x)            (V3 / Kimi-K3)
                    sqrt(Softplus(W_r x))     (DeepSeek-V4)
  * top-k selection over BIASED scores s + b, gate weights from the raw scores
    renormalized over the selected experts (aux-loss-free balancing, V3 lineage)
  * balancers for b:
      "bias"     -- b_j += gamma * sign(mean_load - load_j)      (V3/V4)
      "quantile" -- Kimi-K3 Quantile Balancing: with per-token selection
                    threshold alpha_i (k-th largest biased score), set
                    b_j = -quantile_{1-k/n}(s_{:,j} - alpha) so each expert is
                    selected by ~k/n of tokens; recenter, EMA-smooth. One
                    forward pass, no auxiliary loss, no sign-update lag.
  * V4's small sequence-wise balance loss (prevents within-sequence collapse)
  * V4 hash routing option for early layers (expert = hash(token_id))
  * K3 Stable LatentMoE: routed experts operate in a narrow latent space
        u = sum_i p_i E_i(W_down x);  y = sum_j Shared_j(x) + W_up RMSNorm(u)
    decoupling expert/router cost from model width (what makes 896-expert
    top-16 affordable). Set moe_latent_dim=None for V4-style full-width experts.
  * SiTU-GLU or SwiGLU expert activation
  * QAT: MXFP4 fake-quant on expert weights, MXFP8 on expert activations (K3)
  * expert compute runs as ONE grouped GEMM (Triton kernel on GPU)
  * `dispatcher`: optional DeepEP-style expert-parallel Buffer; when set,
    tokens are exchanged across ranks around the expert compute.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.grouped_gemm import grouped_gemm
from ..kernels.mxfp import mx_fake_quant
from .activations import glu_act
from .norm import RMSNorm

_HASH_P, _HASH_Q = 2654435761, 40503


class DenseFFN(nn.Module):
    def __init__(self, dim: int, hidden: int, act: str = "swiglu",
                 beta1: float = 4.0, beta2: float = 25.0):
        super().__init__()
        self.w_in = nn.Linear(dim, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, dim, bias=False)
        self.act, self.b1, self.b2 = act, beta1, beta2

    def forward(self, x):
        return self.w_out(glu_act(self.w_in(x), self.act, self.b1, self.b2))


class LatentMoE(nn.Module):
    def __init__(self, dim: int, n_experts: int, top_k: int, expert_hidden: int,
                 n_shared: int = 1, shared_hidden: int = 512,
                 latent_dim: int | None = None, score_fn: str = "sigmoid",
                 balancer: str = "bias", bias_update_rate: float = 1e-2,
                 quantile_ema: float = 0.9, seq_balance_coef: float = 1e-4,
                 act: str = "swiglu", situ_beta1: float = 4.0, situ_beta2: float = 25.0,
                 hash_route: bool = False, qat: bool = False, eps: float = 1e-5,
                 init_std: float = 0.02):
        super().__init__()
        self.e, self.k = n_experts, top_k
        self.score_fn, self.balancer = score_fn, balancer
        self.gamma, self.q_ema = bias_update_rate, quantile_ema
        self.seq_coef = seq_balance_coef
        self.hash_route, self.qat = hash_route, qat
        self.act, self.b1, self.b2 = act, situ_beta1, situ_beta2

        d_in = latent_dim if latent_dim is not None else dim
        self.d_in = d_in
        self.w_router = nn.Linear(dim, n_experts, bias=False)
        self.w_router.weight._no_muon = True
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        self.register_buffer("_load_acc", torch.zeros(n_experts))

        if latent_dim is not None:
            self.w_down = nn.Linear(dim, latent_dim, bias=False)
            self.u_norm = RMSNorm(latent_dim, eps)
            self.w_up = nn.Linear(latent_dim, dim, bias=False)
        else:
            self.w_down = self.u_norm = self.w_up = None

        # routed experts as stacked weights -> one grouped GEMM per matmul
        self.w1 = nn.Parameter(torch.randn(n_experts, d_in, 2 * expert_hidden) * init_std)
        self.w2 = nn.Parameter(torch.randn(n_experts, expert_hidden, d_in) * init_std)
        self.shared = (
            DenseFFN(dim, shared_hidden * n_shared, act, situ_beta1, situ_beta2)
            if n_shared > 0 else None
        )
        self.dispatcher = None  # DeepEP-style Buffer, set by the trainer for EP
        # optional process group for GLOBAL balancer statistics under EP
        # (papers aggregate load globally; default None = per-rank stats).
        # Set to dist.group.WORLD (pure EP) or the EP row group (2D).
        self.stats_group = None

    # ------------------------------------------------------------------ #
    def _scores(self, flat: torch.Tensor) -> torch.Tensor:
        # routing math always in fp32 (bias balancing is sensitive to precision)
        logits = F.linear(flat.float(), self.w_router.weight.float())
        if self.score_fn == "sigmoid":
            return torch.sigmoid(logits)
        return torch.sqrt(F.softplus(logits))  # V4: sqrt(softplus)

    @torch.no_grad()
    def _quantile_balance(self, scores: torch.Tensor):
        """Kimi-K3 QB: set bias from the (1-k/n)-quantile of routing margins."""
        biased = scores + self.expert_bias
        alpha = biased.topk(self.k, dim=-1).values[:, -1:]  # per-token threshold
        margins = scores - alpha                            # [N, E]
        q = 1.0 - self.k / self.e
        b_hat = -torch.quantile(margins, q, dim=0)
        if self.stats_group is not None:
            # global stats: mean of per-rank quantiles across the EP group
            # (approximates the global quantile; keeps balancer state
            # identical on all group members). Collective ordering is safe:
            # every EP-group member runs the same layers in the same order.
            dist.all_reduce(b_hat, group=self.stats_group)
            b_hat.div_(dist.get_world_size(self.stats_group))
        b_hat = b_hat - b_hat.mean()
        self.expert_bias.mul_(self.q_ema).add_(b_hat, alpha=1.0 - self.q_ema)

    @torch.no_grad()
    def balance_step(self):
        """V3/V4 aux-loss-free bias update; call once per optimizer step."""
        if self.balancer != "bias":
            return
        if self.stats_group is not None:
            # all-reduce BEFORE the zero-load early-out so every group member
            # takes the same branch (collective-safe), and the sign update
            # sees the exact GLOBAL load counts.
            dist.all_reduce(self._load_acc, group=self.stats_group)
        if self._load_acc.sum() == 0:
            return
        load = self._load_acc / self._load_acc.sum()
        self.expert_bias.add_(self.gamma * torch.sign(1.0 / self.e - load))
        self._load_acc.zero_()

    # ------------------------------------------------------------------ #
    def _route(self, flat: torch.Tensor, token_ids: torch.Tensor | None, shape):
        N = flat.shape[0]
        if self.hash_route:
            assert token_ids is not None, "hash routing needs token ids"
            tid = token_ids.reshape(-1, 1)
            slots = torch.arange(self.k, device=flat.device).view(1, -1)
            sel = ((tid * _HASH_P + slots * _HASH_Q) % self.e).long()
            weights = torch.full((N, self.k), 1.0 / self.k, device=flat.device)
            return sel, weights, flat.new_zeros(())
        scores = self._scores(flat)
        if self.training and self.balancer == "quantile":
            self._quantile_balance(scores)
        biased = scores + self.expert_bias
        _, sel = torch.topk(biased, self.k, dim=-1)
        w = scores.gather(-1, sel)
        weights = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        seq_loss = flat.new_zeros(())
        if self.training:
            with torch.no_grad():
                counts = torch.bincount(sel.reshape(-1), minlength=self.e).float()
                self._load_acc += counts
            if self.seq_coef > 0:
                B, T = shape
                sel_bt = sel.view(B, T, self.k)
                p = (scores / scores.sum(-1, keepdim=True).clamp_min(1e-9)).view(B, T, self.e)
                f = torch.zeros(B, self.e, device=flat.device)
                ones = torch.ones_like(sel_bt, dtype=f.dtype).reshape(B, -1)
                f.scatter_add_(1, sel_bt.reshape(B, -1), ones)
                f = f * self.e / (T * self.k)
                seq_loss = self.seq_coef * (f.detach() * p.mean(dim=1)).sum(-1).mean()
        return sel, weights.to(flat.dtype), seq_loss

    def _expert_ffn(self, x_sorted: torch.Tensor, group_sizes: torch.Tensor) -> torch.Tensor:
        w1, w2 = self.w1, self.w2
        if self.dispatcher is not None:  # EP: this rank computes only its expert slice
            sl = self.dispatcher.expert_slice
            w1, w2 = w1[sl], w2[sl]
        if self.qat:
            w1 = mx_fake_quant(w1, "fp4_e2m1")
            w2 = mx_fake_quant(w2, "fp4_e2m1")
            x_sorted = mx_fake_quant(x_sorted, "fp8_e4m3")
        h = glu_act(grouped_gemm(x_sorted, w1, group_sizes), self.act, self.b1, self.b2)
        if self.qat:
            h = mx_fake_quant(h, "fp8_e4m3")
        return grouped_gemm(h, w2, group_sizes)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, token_ids: torch.Tensor | None = None):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        sel, weights, seq_loss = self._route(flat, token_ids, (B, T))

        xin = self.w_down(flat) if self.w_down is not None else flat

        if self.dispatcher is not None:
            u = self.dispatcher.moe_forward(xin, sel, weights, self._expert_ffn)
        else:
            flat_sel = sel.reshape(-1)
            order = torch.argsort(flat_sel, stable=True)
            group_sizes = torch.bincount(flat_sel, minlength=self.e)
            tok_of = order // self.k
            y_sorted = self._expert_ffn(xin[tok_of], group_sizes)
            w_sorted = weights.reshape(-1)[order].unsqueeze(-1)
            u = torch.zeros_like(xin)
            u.index_add_(0, tok_of, (y_sorted * w_sorted).to(u.dtype))

        if self.w_up is not None:
            y = self.w_up(self.u_norm(u))
        else:
            y = u
        if self.shared is not None:
            y = y + self.shared(flat)

        aux = {"seq_balance_loss": seq_loss,
               "load": (self._load_acc / self._load_acc.sum().clamp_min(1)).detach()}
        return y.view(B, T, D), aux
