"""Compressed Attention (DeepSeek-V4): CSA and HCA.

Shared-KV MQA over *compressed* entries, with query compression and grouped
output projection. Per token we cache one entry

    e_t = [ c_t ; k^R_t ],   c_t = RMSNorm(W_dkv x_t)  (kv_latent_dim, "FP8 half")
                             k^R_t = RoPE(W_kr x_t)    (rope_dim,      "BF16 half")

and *compress* completed blocks of entries with learned, per-dimension
saliency weights (softmax over the block rows of Z + learnable positional
bias, Z = W_z c):

  CSA ("Compressed Sparse Attention", block m, OVERLAPPED):
      entry i = softmax-weighted sum over tokens [m(i-1), m(i+1))  (2m rows,
      two bias tables), then a tiny FP4-friendly *lightning indexer* scores
      every (query, compressed entry) pair and each query attends to only its
      top-k compressed entries -- sparse selection over an already-compressed
      cache.
  HCA ("Heavily Compressed Attention", block m' >> m):
      entry i = softmax-weighted sum over tokens [m'i, m'(i+1)), and queries
      attend DENSELY over the short compressed cache.

Every query additionally sees the raw (uncompressed) causal tail of
`local_window` tokens, which also covers the not-yet-completed block.
KV-cache cost per token: D_e/m floats amortized + a fixed raw tail -- the
"10% KV cache / 27% FLOPs at 1M tokens" mechanism of the V4 report.

Queries are low-rank (W_dq -> RMSNorm -> W_uq, one more bottleneck), values
are the latent part c of each entry, and head outputs are projected through
V4's grouped output projection (g groups -> d_g each) before W_o.

The training path is fully vectorized; the inference cache builds compressed
entries incrementally as blocks complete and keeps only {compressed entries,
indexer keys, raw tail} -- the unit tests check bit-consistency of the two.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..kernels.indexer import indexer_scores
from .norm import RMSNorm
from .rope import RotaryEmbedding

NEG_INF = float("-inf")


class CompressedAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, mode: str,
                 kv_latent_dim: int = 128, q_latent_dim: int = 192, rope_dim: int = 32,
                 csa_block: int = 16, csa_topk: int = 16, hca_block: int = 64,
                 local_window: int = 128, indexer_heads: int = 4, indexer_dim: int = 32,
                 out_groups: int = 2, out_group_dim: int = 128,
                 rope_theta: float = 10000.0, rope_scaling: float = 1.0,
                 eps: float = 1e-5, fp4_indexer_sim: bool = False):
        super().__init__()
        assert mode in ("csa", "hca")
        self.mode = mode
        self.h = n_heads
        self.d_c, self.d_r = kv_latent_dim, rope_dim
        self.d_e = kv_latent_dim + rope_dim
        self.block = csa_block if mode == "csa" else hca_block
        self.topk = csa_topk
        self.window = local_window
        self.fp4_indexer_sim = fp4_indexer_sim
        if mode == "csa":
            assert local_window >= 2 * self.block, "raw tail must cover the CSA overlap"
        else:
            assert local_window >= self.block

        self.w_dkv = nn.Linear(dim, kv_latent_dim, bias=False)
        self.kv_norm = RMSNorm(kv_latent_dim, eps)
        self.w_kr = nn.Linear(dim, rope_dim, bias=False)
        self.w_dq = nn.Linear(dim, q_latent_dim, bias=False)
        self.q_norm = RMSNorm(q_latent_dim, eps)
        self.w_uq = nn.Linear(q_latent_dim, n_heads * self.d_e, bias=False)
        self.w_uq.weight._muon_heads = n_heads  # K3-style per-head Muon
        self.rope = RotaryEmbedding(rope_dim, rope_theta, rope_scaling)

        # learned compression: per-dim saliency logits + positional biases
        self.w_z = nn.Linear(kv_latent_dim, self.d_e, bias=False)
        if mode == "csa":
            self.bias_a = nn.Parameter(torch.zeros(self.block, self.d_e))
            self.bias_b = nn.Parameter(torch.zeros(self.block, self.d_e))
            self.bias_a._no_muon = self.bias_b._no_muon = True
            # lightning indexer
            self.hi, self.di = indexer_heads, indexer_dim
            self.w_iq = nn.Linear(q_latent_dim, indexer_heads * indexer_dim, bias=False)
            self.w_iw = nn.Linear(q_latent_dim, indexer_heads, bias=False)
            self.w_ik = nn.Linear(self.d_e, indexer_dim, bias=False)
        else:
            self.bias = nn.Parameter(torch.zeros(self.block, self.d_e))
            self.bias._no_muon = True

        # grouped output projection: values are the d_c latent part
        assert n_heads % out_groups == 0
        self.g, self.d_g = out_groups, out_group_dim
        self.w_gout = nn.Parameter(
            torch.randn(out_groups, (n_heads // out_groups) * kv_latent_dim, out_group_dim)
            * (1.0 / math.sqrt((n_heads // out_groups) * kv_latent_dim))
        )
        self.wo = nn.Linear(out_groups * out_group_dim, dim, bias=False)

        self.track_qk = False
        self.register_buffer("max_qk_logit", torch.zeros(n_heads), persistent=False)

    # ------------------------------------------------------------------ #
    # entry construction & compression
    # ------------------------------------------------------------------ #
    def _entries(self, x: torch.Tensor, positions: torch.Tensor):
        c = self.kv_norm(self.w_dkv(x))
        kr = self.rope.rotate(self.w_kr(x), positions)
        e = torch.cat([c, kr], dim=-1)          # [B, T, D_e]
        z = self.w_z(c)                          # saliency logits [B, T, D_e]
        return e, z

    def _compress(self, e: torch.Tensor, z: torch.Tensor, first_entry: int, n_entries: int,
                  base_pos: int):
        """Compress entries [first_entry, first_entry+n_entries) from raw (e, z).

        Raw row j of `e` holds global token (base_pos + j). Entry i covers
        global tokens [m(i-1), m(i+1)) for CSA, [m i, m(i+1)) for HCA.
        Returns [B, n_entries, D_e].
        """
        if n_entries <= 0:
            return e.new_zeros(e.shape[0], 0, self.d_e)
        B = e.shape[0]
        m = self.block
        idx_i = torch.arange(first_entry, first_entry + n_entries, device=e.device)
        offs = torch.arange(m, device=e.device)

        def rows(start_of_block):  # [nE, m] global token ids -> local rows
            tok = start_of_block[:, None] + offs[None, :]
            local = tok - base_pos
            valid = (tok >= 0) & (local >= 0)
            return local.clamp(0, e.shape[1] - 1), valid

        la, va = rows(idx_i * m)                          # window a: [m i, m(i+1))
        ea = e[:, la]                                      # [B, nE, m, D_e]
        za = z[:, la] + self.bias_a if self.mode == "csa" else z[:, la] + self.bias
        za = za.masked_fill(~va[None, :, :, None], NEG_INF)
        if self.mode == "csa":
            lb, vb = rows((idx_i - 1) * m)                # window b: [m(i-1), m i)
            eb = e[:, lb]
            zb = (z[:, lb] + self.bias_b).masked_fill(~vb[None, :, :, None], NEG_INF)
            allz = torch.cat([za, zb], dim=2)             # [B, nE, 2m, D_e]
            alle = torch.cat([ea, eb], dim=2)
        else:
            allz, alle = za, ea
        S = torch.softmax(allz.float(), dim=2).to(e.dtype)
        return torch.einsum("bnmd,bnmd->bnd", S, alle)

    # ------------------------------------------------------------------ #
    # attention over [selected/dense compressed entries] + [raw tail]
    # ------------------------------------------------------------------ #
    def _attend(self, x, cq, q, q_pos, comp, raw, raw_base):
        """q: [B,T,H,D_e]; q_pos: [T] global; comp: [B,nE,D_e] (entry i ends at
        global token m(i+1)-1); raw: [B,L,D_e] holding globals [raw_base, ...)."""
        B, T, H, D_e = q.shape
        m, W = self.block, self.window
        nE = comp.shape[1]
        scale = 1.0 / math.sqrt(D_e)

        # -------- compressed-entry logits --------
        if nE > 0:
            ends = (torch.arange(nE, device=q.device) + 1) * m       # exclusive end
            allowed = ends[None, :] <= (q_pos[:, None] + 1)          # [T, nE]
            if self.mode == "csa":
                kI = self.w_ik(comp)                                  # [B, nE, DI]
                qI = self.w_iq(cq).view(B, T, self.hi, self.di)
                wI = self.w_iw(cq)
                scores = indexer_scores(qI, wI, kI, fp4_sim=self.fp4_indexer_sim)
                scores = scores.masked_fill(~allowed[None], NEG_INF)  # [B, T, nE]
                k_eff = min(self.topk, nE)
                sel_val, sel = torch.topk(scores, k_eff, dim=-1)      # [B, T, k]
                sel_comp = comp.gather(1, sel.reshape(B, -1, 1).expand(-1, -1, D_e))
                sel_comp = sel_comp.view(B, T, k_eff, D_e)
                # an entry is attended only if allowed AND the indexer scored
                # it positive -- ReLU'd scores tie at exactly 0 ("irrelevant"),
                # and dropping those makes selection independent of topk
                # tie-breaking (=> the incremental cache is bit-consistent
                # with the full forward)
                sel_ok = allowed[None].expand(B, -1, -1).gather(2, sel)  # [B, T, k]
                sel_ok = sel_ok & (sel_val > 0)
                lc = torch.einsum("bthd,btkd->bthk", q, sel_comp) * scale
                lc = lc.masked_fill(~sel_ok[:, :, None, :], NEG_INF)
                vals_c = sel_comp[..., : self.d_c]                    # [B, T, k, d_c]
                per_query_comp = True
            else:
                lc = torch.einsum("bthd,bnd->bthn", q, comp) * scale
                lc = lc.masked_fill(~allowed[None, :, None, :], NEG_INF)
                vals_c = comp[..., : self.d_c]                        # [B, nE, d_c]
                per_query_comp = False
        else:
            lc = None

        # -------- raw tail logits (last W tokens per query) --------
        L = raw.shape[1]
        w_off = torch.arange(W, device=q.device)
        tok = q_pos[:, None] - W + 1 + w_off[None, :]                 # [T, W] global ids
        local = tok - raw_base
        ok = (tok >= 0) & (local >= 0) & (local < L)
        local_c = local.clamp(0, max(L - 1, 0))
        raw_g = raw[:, local_c.reshape(-1)].view(B, T, W, D_e)
        lr = torch.einsum("bthd,btwd->bthw", q, raw_g) * scale
        lr = lr.masked_fill(~ok[None, :, None, :], NEG_INF)
        vals_r = raw_g[..., : self.d_c]

        # -------- joint softmax --------
        if lc is None:
            logits = lr
        else:
            logits = torch.cat([lc, lr], dim=-1)
        if self.track_qk and self.training:
            with torch.no_grad():
                fin = logits.float().masked_fill(torch.isinf(logits.float()), 0.0)
                self.max_qk_logit = fin.abs().amax(dim=(0, 1, 3))
        p = torch.softmax(logits.float(), dim=-1).to(q.dtype)

        if lc is None:
            o = torch.einsum("bthw,btwc->bthc", p, vals_r)
        else:
            nc = lc.shape[-1]
            pc, pr = p[..., :nc], p[..., nc:]
            if per_query_comp:
                o = torch.einsum("bthk,btkc->bthc", pc, vals_c)
            else:
                o = torch.einsum("bthn,bnc->bthc", pc, vals_c)
            o = o + torch.einsum("bthw,btwc->bthc", pr, vals_r)

        # -------- grouped output projection --------
        o = o.reshape(B, T, self.g, (H // self.g) * self.d_c)
        o = torch.einsum("btgi,gio->btgo", o, self.w_gout.to(o.dtype))
        return self.wo(o.reshape(B, T, self.g * self.d_g))

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, cache: dict | None = None, positions=None) -> torch.Tensor:
        B, T, _ = x.shape
        m = self.block
        if cache is None:
            pos0 = 0
            positions = torch.arange(T, device=x.device) if positions is None else positions
            e, z = self._entries(x, positions)
            n_entries = T // m
            comp = self._compress(e, z, 0, n_entries, base_pos=0)
            cq = self.q_norm(self.w_dq(x))
            q = self.w_uq(cq).view(B, T, self.h, self.d_e)
            qc, qr = q.split([self.d_c, self.d_r], dim=-1)
            q = torch.cat([qc, self.rope.rotate(qr, positions)], dim=-1)
            return self._attend(x, cq, q, positions, comp, e, raw_base=0)

        # ---------------- incremental (inference) path ----------------
        pos0 = cache.get("pos", 0)
        positions = torch.arange(pos0, pos0 + T, device=x.device) if positions is None else positions
        e_new, z_new = self._entries(x, positions)
        raw_e = torch.cat([cache["raw_e"], e_new], dim=1) if "raw_e" in cache else e_new
        raw_z = torch.cat([cache["raw_z"], z_new], dim=1) if "raw_z" in cache else z_new
        raw_base = cache.get("raw_base", 0)
        total = pos0 + T

        n_built = cache.get("n_comp", 0)
        n_total = total // m
        new_comp = self._compress(raw_e, raw_z, n_built, n_total - n_built, base_pos=raw_base)
        comp = torch.cat([cache["comp"], new_comp], dim=1) if "comp" in cache else new_comp

        cq = self.q_norm(self.w_dq(x))
        q = self.w_uq(cq).view(B, T, self.h, self.d_e)
        qc, qr = q.split([self.d_c, self.d_r], dim=-1)
        q = torch.cat([qc, self.rope.rotate(qr, positions)], dim=-1)
        out = self._attend(x, cq, q, positions, comp, raw_e, raw_base=raw_base)

        # trim the raw tail AFTER attending: keep only what future compression
        # windows and future queries' local windows still need
        keep_from = max(0, total - max(self.window, 2 * m + (total % m)))
        keep_from = min(keep_from, raw_base + raw_e.shape[1])
        if keep_from > raw_base:
            raw_e = raw_e[:, keep_from - raw_base:]
            raw_z = raw_z[:, keep_from - raw_base:]
            raw_base = keep_from
        cache.update(
            raw_e=raw_e.detach(), raw_z=raw_z.detach(), raw_base=raw_base,
            comp=comp.detach(), n_comp=n_total, pos=total,
        )
        return out

    @torch.no_grad()
    def qk_clip_(self, tau: float):
        """QK-clip for shared-KV MQA: keys are shared, so the full tau/max
        factor lands on the per-head query up-projection."""
        for h in range(self.h):
            s_max = float(self.max_qk_logit[h])
            if s_max > tau:
                rows = slice(h * self.d_e, (h + 1) * self.d_e)
                self.w_uq.weight[rows].mul_(tau / s_max)
