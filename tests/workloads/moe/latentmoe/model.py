"""LatentMoE transformer: runtime-selectable DeepSeek-V4 / Kimi-K3 / hybrid stack.

Per-layer plan comes from the config:
  attention:  csa | hca            (V4 compressed attention)
              kda | mla            (K3: 3 KDA : 1 Gated MLA, NoPE)
  ffn:        dense | hash_moe | moe
  residual:   standard | mhc (V4 hyper-connections) | attnres (K3)

plus a Multi-Token-Prediction head trained to predict t+2 (and usable for
self-speculative decoding).

forward() returns a dict with the LM loss, MTP loss, MoE sequence-balance
loss, per-expert load stats and (optionally) logits. Inference uses per-layer
cache dicts (latent KV / compressed entries / KDA state -- all tiny compared
to a dense KV cache).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import (
    AttentionResidual,
    CompressedAttention,
    DenseFFN,
    GatedMLA,
    HyperConnections,
    KDAAttention,
    LatentMoE,
    MTPHead,
    RMSNorm,
)


def _make_attention(cfg: ModelConfig, kind: str) -> nn.Module:
    if kind in ("csa", "hca"):
        return CompressedAttention(
            cfg.dim, cfg.n_heads, kind,
            kv_latent_dim=cfg.kv_latent_dim, q_latent_dim=cfg.q_latent_dim,
            rope_dim=cfg.rope_dim, csa_block=cfg.csa_block, csa_topk=cfg.csa_topk,
            hca_block=cfg.hca_block, local_window=cfg.local_window,
            indexer_heads=cfg.indexer_heads, indexer_dim=cfg.indexer_dim,
            out_groups=cfg.out_groups, out_group_dim=cfg.out_group_dim,
            rope_theta=cfg.rope_theta, rope_scaling=cfg.rope_scaling,
            eps=cfg.norm_eps, fp4_indexer_sim=cfg.qat_experts,
        )
    if kind == "kda":
        return KDAAttention(cfg.dim, cfg.n_heads, cfg.kda_head_dim,
                            conv_size=cfg.kda_conv, gmin=cfg.kda_gmin,
                            chunk=cfg.kda_chunk, eps=cfg.norm_eps)
    if kind == "mla":
        return GatedMLA(cfg.dim, cfg.n_heads, cfg.head_dim,
                        kv_latent_dim=cfg.kv_latent_dim, q_latent_dim=cfg.q_latent_dim,
                        eps=cfg.norm_eps, dropout=cfg.dropout)
    raise ValueError(kind)


def _make_ffn(cfg: ModelConfig, kind: str) -> nn.Module:
    if kind == "dense":
        return DenseFFN(cfg.dim, cfg.dense_ffn_hidden, cfg.moe_act,
                        cfg.situ_beta1, cfg.situ_beta2)
    return LatentMoE(
        cfg.dim, cfg.n_routed_experts, cfg.top_k, cfg.expert_hidden,
        n_shared=cfg.n_shared_experts, shared_hidden=cfg.shared_expert_hidden,
        latent_dim=cfg.moe_latent_dim, score_fn=cfg.router_score,
        balancer=("none" if kind == "hash_moe" else cfg.balancer),
        bias_update_rate=cfg.bias_update_rate, quantile_ema=cfg.quantile_ema,
        seq_balance_coef=(0.0 if kind == "hash_moe" else cfg.seq_balance_coef),
        act=cfg.moe_act, situ_beta1=cfg.situ_beta1, situ_beta2=cfg.situ_beta2,
        hash_route=(kind == "hash_moe"), qat=cfg.qat_experts, eps=cfg.norm_eps,
        init_std=cfg.init_std,
    )


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, attn_kind: str, ffn_kind: str):
        super().__init__()
        self.attn_kind, self.ffn_kind = attn_kind, ffn_kind
        self.residual = cfg.residual
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = _make_attention(cfg, attn_kind)
        self.ffn = _make_ffn(cfg, ffn_kind)
        if cfg.residual == "mhc":
            self.hc_attn = HyperConnections(cfg.mhc_streams, cfg.sinkhorn_iters)
            self.hc_ffn = HyperConnections(cfg.mhc_streams, cfg.sinkhorn_iters)
        elif cfg.residual == "attnres":
            self.attnres = AttentionResidual(cfg.dim)

    def _ffn(self, x, token_ids):
        if isinstance(self.ffn, LatentMoE):
            return self.ffn(x, token_ids)
        return self.ffn(x), None

    def forward(self, state, cache, positions, token_ids, sources):
        """state: [B,T,d] (standard/attnres) or [B,T,n,d] streams (mhc)."""
        aux = None
        if self.residual == "mhc":
            X = state
            y = self.attn(self.attn_norm(self.hc_attn.read(X)), cache=cache, positions=positions)
            X = self.hc_attn.write(X, y)
            y, aux = self._ffn(self.ffn_norm(self.hc_ffn.read(X)), token_ids)
            X = self.hc_ffn.write(X, y)
            return X, aux
        h = state
        if self.residual == "attnres" and sources:
            h = h + self.attnres(sources)
        h = h + self.attn(self.attn_norm(h), cache=cache, positions=positions)
        y, aux = self._ffn(self.ffn_norm(h), token_ids)
        return h + y, aux


class LatentMoEModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.embed.weight._no_muon = True
        attn_plan, ffn_plan = cfg.attn_plan(), cfg.ffn_plan()
        self.blocks = nn.ModuleList(
            Block(cfg, a, f) for a, f in zip(attn_plan, ffn_plan)
        )
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
            self.lm_head.weight._no_muon = True
        self.mtp = (
            MTPHead(cfg.dim, cfg.n_heads, cfg.dense_ffn_hidden,
                    fuse_layers=cfg.mtp_fuse_layers, eps=cfg.norm_eps)
            if cfg.use_mtp else None
        )
        # feature taps for K3-style MTP fusion: early / middle / last block
        n = cfg.n_layers
        self._taps = (max(0, n // 4 - 1), n // 2, n - 1)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.cfg.init_std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    # ------------------------------------------------------------------ #
    def head(self, h: torch.Tensor) -> torch.Tensor:
        w = self.embed.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(h, w)

    def new_caches(self):
        return [dict() for _ in self.blocks]

    @torch.no_grad()
    def fork_caches(self, caches):
        """Deep snapshot of the per-layer caches (all value tensors cloned).

        Speculative decoding uses this to REWIND: snapshot before the
        draft+verify forward, restore on rejection, so the cache never
        retains a rejected draft token. Restore with:
            for c, s in zip(caches, snapshot): c.clear(); c.update(s)
        """
        return [{k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
                 for k, v in c.items()} for c in caches]

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None,
                caches: list | None = None, positions: torch.Tensor | None = None,
                return_logits: bool = True):
        cfg = self.cfg
        B, T = tokens.shape
        h = self.embed(tokens)
        emb = h
        use_mhc = cfg.residual == "mhc"
        state = HyperConnections.expand(h, cfg.mhc_streams) if use_mhc else h

        sources = [emb]          # AttnRes sources (embedding + recent blocks)
        feats = {}
        seq_losses, loads = [], []
        for i, blk in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            state, aux = blk(state, cache, positions, tokens, list(sources))
            if aux is not None:
                seq_losses.append(aux["seq_balance_loss"])
                loads.append(aux["load"])
            h_i = HyperConnections.collapse(state) if use_mhc else state
            if cfg.residual == "attnres":
                sources.append(h_i)
                if len(sources) > 1 + cfg.attnres_block:
                    sources = [sources[0]] + sources[-cfg.attnres_block:]
            if i in self._taps:
                feats[i] = h_i

        h = HyperConnections.collapse(state) if use_mhc else state
        h_final = self.final_norm(h)

        out = {}
        logits = None
        if return_logits or targets is not None:
            logits = self.head(h_final)
            if return_logits:
                out["logits"] = logits

        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, cfg.vocab_size).float(),
                                 targets.reshape(-1), ignore_index=-100)
            loss = ce
            out["ce_loss"] = ce
            if self.mtp is not None and T > 2:
                # predict token t+2 from h_t and emb(target_t = token t+1)
                next_tok = targets[:, :-1].clamp_min(0)          # token t+1
                mtp_h = self.mtp(h[:, :-1], self.embed(next_tok),
                                 feats=[f[:, :-1] for f in
                                        (feats[j] for j in self._taps)] if cfg.mtp_fuse_layers else None)
                mtp_logits = self.head(mtp_h)                    # predicts t+2
                mtp_targets = targets[:, 1:]
                mtp_loss = F.cross_entropy(
                    mtp_logits.reshape(-1, cfg.vocab_size).float(),
                    mtp_targets.reshape(-1), ignore_index=-100)
                loss = loss + cfg.mtp_loss_weight * mtp_loss
                out["mtp_loss"] = mtp_loss
            if seq_losses:
                sb = torch.stack(seq_losses).sum()
                loss = loss + sb
                out["seq_balance_loss"] = sb
            out["loss"] = loss
        if loads:
            out["expert_load"] = torch.stack(loads).mean(0)
        out["hidden"] = h_final
        out["hidden_prenorm"] = h
        if self.mtp is not None and cfg.mtp_fuse_layers:
            out["mtp_feats"] = [feats[j] for j in self._taps]
        return out

    # ------------------------------------------------------------------ #
    # trainer hooks
    # ------------------------------------------------------------------ #
    def moe_modules(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, LatentMoE)]

    def balance_step(self):
        for m in self.moe_modules():
            m.balance_step()

    def set_track_qk(self, flag: bool):
        for m in self.modules():
            if hasattr(m, "track_qk"):
                m.track_qk = flag

    @torch.no_grad()
    def apply_qk_clip(self, tau: float = 100.0):
        for m in self.modules():
            if hasattr(m, "qk_clip_"):
                m.qk_clip_(tau)

    def set_dispatcher(self, buffer):
        for m in self.moe_modules():
            if not m.hash_route:
                m.dispatcher = buffer
