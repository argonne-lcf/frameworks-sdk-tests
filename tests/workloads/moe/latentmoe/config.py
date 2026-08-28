"""Model configuration.

One config drives both architecture families; ``arch`` presets select them at runtime:

  * ``"v4"``     -- DeepSeek-V4-Pro style   (arXiv:2606.19348)
        CSA/HCA compressed attention, mHC hyper-connections, sqrt-softplus
        routing with aux-loss-free bias balancing (+ small sequence-wise loss),
        hash routing in early layers, MTP, Muon-friendly parameter layout.
  * ``"k3"``     -- Kimi-K3 style           (arXiv:2607.24653)
        KDA linear attention interleaved 3:1 with Gated MLA (NoPE),
        Attention Residuals, Stable LatentMoE (latent-space experts,
        sigmoid router + Quantile Balancing, SiTU-GLU), MTP.
  * ``"hybrid"`` -- KDA for most layers + CSA for the global layers, LatentMoE
        with V4's sqrt-softplus router. A demonstration that the pieces compose.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelConfig:
    # ---- core dims ------------------------------------------------------
    vocab_size: int = 8192
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    head_dim: int = 64
    max_seq_len: int = 4096
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    # ---- attention architecture ----------------------------------------
    # per-layer pattern is derived from these knobs (see layer_plan()):
    #   "csa" | "hca" | "kda" | "mla"
    attention: str = "csa_hca"            # "csa_hca" (V4) | "kda_mla" (K3) | "kda_csa" (hybrid)
    # V4 compressed attention -------------------------------------------
    kv_latent_dim: int = 128               # c: shared compressed KV entry width (MQA-style)
    q_latent_dim: int = 192                 # d_c: query down-projection width
    rope_dim: int = 32                      # decoupled RoPE dims (BF16 half of the KV cache)
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0               # NTK-by-parts style scale for long context
    csa_block: int = 16                     # m : tokens per CSA compressed entry (overlapped)
    csa_topk: int = 16                      # top-k compressed entries per query (lightning indexer)
    hca_block: int = 64                     # m': tokens per HCA compressed entry (dense attn after)
    local_window: int = 128                 # raw (uncompressed) causal tail every query also sees
    indexer_heads: int = 4                  # lightning indexer heads
    indexer_dim: int = 32                   # lightning indexer head dim
    hca_every: int = 4                      # every hca_every-th attention layer uses HCA, rest CSA
    out_groups: int = 2                     # grouped output projection (g groups -> d_g each)
    out_group_dim: int = 128                # d_g
    # K3 attention --------------------------------------------------------
    kda_head_dim: int = 64                  # KDA key/query head dim (dk); value dim = dk
    kda_chunk: int = 16                     # chunkwise scan tile (paper: 16-token tiles)
    kda_gmin: float = -5.0                  # lower bound of log-decay (alpha > e^-5)
    kda_conv: int = 4                       # short-conv kernel size on q/k/v
    mla_every: int = 4                      # 1 Gated-MLA per mla_every layers (3 KDA : 1 MLA)

    # ---- residual scheme -------------------------------------------------
    residual: str = "standard"              # "standard" | "mhc" | "attnres"
    mhc_streams: int = 4                    # n_hc residual streams
    sinkhorn_iters: int = 20                # Birkhoff projection iterations (paper: t_max=20)
    attnres_block: int = 4                  # AttnRes block-variant size (paper: ~12; small here)

    # ---- MoE -------------------------------------------------------------
    moe_layers_start: int = 1               # first N layers use dense FFN (or hash routing)
    n_routed_experts: int = 32
    n_shared_experts: int = 1
    top_k: int = 4
    moe_latent_dim: Optional[int] = 256     # K3 Stable LatentMoE: experts live in this width.
    #                                         None => experts operate on full `dim` (V3/V4 style)
    expert_hidden: int = 256                # per-expert FFN hidden ("fine-grained" experts)
    shared_expert_hidden: int = 512
    dense_ffn_hidden: int = 1024
    router_score: str = "sigmoid"           # "sigmoid" (K3/V3) | "sqrt_softplus" (V4)
    balancer: str = "bias"                  # "bias" (V3/V4 aux-free) | "quantile" (K3) | "none"
    bias_update_rate: float = 1e-2          # gamma for "bias" balancer
    quantile_ema: float = 0.9               # EMA for "quantile" balancer bias
    seq_balance_coef: float = 1e-4          # V4: small sequence-wise balance loss
    hash_routing_layers: int = 0            # V4: first K MoE layers route by token-id hash
    moe_act: str = "swiglu"                 # "swiglu" | "situ_glu"
    situ_beta1: float = 4.0                 # SiTU-GLU softcap on gate branch
    situ_beta2: float = 25.0                # SiTU-GLU softcap on up branch
    # quantization-aware training (Kimi-K3: MXFP4 weights / MXFP8 activations on experts;
    # DeepSeek-V4 stores routed expert params in FP4) -- fake-quant with STE:
    qat_experts: bool = False

    # ---- multi-token prediction ------------------------------------------
    use_mtp: bool = True
    mtp_loss_weight: float = 0.3
    mtp_fuse_layers: bool = True            # K3/EAGLE-3 style low/mid/high feature fusion

    # ---- misc -------------------------------------------------------------
    init_std: float = 0.02
    dropout: float = 0.0

    # ======================================================================
    def __post_init__(self):
        assert self.attention in ("csa_hca", "kda_mla", "kda_csa")
        assert self.residual in ("standard", "mhc", "attnres")
        assert self.balancer in ("bias", "quantile", "none")
        assert self.router_score in ("sigmoid", "sqrt_softplus")
        assert self.moe_act in ("swiglu", "situ_glu")
        assert self.top_k <= self.n_routed_experts
        assert self.local_window >= self.csa_block, "local window must cover the open CSA block"
        assert self.local_window >= self.hca_block, "local window must cover the open HCA block"

    # ---- per-layer plans --------------------------------------------------
    def attn_plan(self) -> list[str]:
        """Attention flavour per layer."""
        plan = []
        for i in range(self.n_layers):
            if self.attention == "csa_hca":
                plan.append("hca" if (i + 1) % self.hca_every == 0 else "csa")
            elif self.attention == "kda_mla":
                plan.append("mla" if (i + 1) % self.mla_every == 0 else "kda")
            else:  # hybrid: KDA everywhere, CSA on the "global" slots
                plan.append("csa" if (i + 1) % self.mla_every == 0 else "kda")
        return plan

    def ffn_plan(self) -> list[str]:
        """FFN flavour per layer: "dense" | "hash_moe" | "moe"."""
        plan = []
        moe_idx = 0
        for i in range(self.n_layers):
            if i < self.moe_layers_start:
                plan.append("dense")
            else:
                plan.append("hash_moe" if moe_idx < self.hash_routing_layers else "moe")
                moe_idx += 1
        return plan

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------
# Presets ("mini" scale: runnable on one GPU / CPU; scale knobs, not structure)
# --------------------------------------------------------------------------

def deepseek_v4_mini(**overrides) -> ModelConfig:
    cfg = dict(
        attention="csa_hca",
        residual="mhc",
        router_score="sqrt_softplus",
        balancer="bias",
        hash_routing_layers=1,
        moe_latent_dim=None,          # V4 experts operate at model width
        moe_act="swiglu",
        use_mtp=True,
        mtp_fuse_layers=False,        # V4 keeps V3's single-stream MTP
    )
    cfg.update(overrides)
    return ModelConfig(**cfg)


def kimi_k3_mini(**overrides) -> ModelConfig:
    cfg = dict(
        attention="kda_mla",
        residual="attnres",
        router_score="sigmoid",
        balancer="quantile",
        moe_latent_dim=256,           # Stable LatentMoE: latent = dim / 2
        moe_act="situ_glu",
        use_mtp=True,
        mtp_fuse_layers=True,
    )
    cfg.update(overrides)
    return ModelConfig(**cfg)


def hybrid_mini(**overrides) -> ModelConfig:
    cfg = dict(
        attention="kda_csa",
        residual="mhc",
        router_score="sqrt_softplus",
        balancer="quantile",
        moe_latent_dim=256,
        moe_act="situ_glu",
        use_mtp=True,
        mtp_fuse_layers=True,
    )
    cfg.update(overrides)
    return ModelConfig(**cfg)


PRESETS = {"v4": deepseek_v4_mini, "k3": kimi_k3_mini, "hybrid": hybrid_mini}


def get_preset(name: str, **overrides) -> ModelConfig:
    try:
        return PRESETS[name](**overrides)
    except KeyError:
        raise KeyError(f"unknown preset '{name}', choose from {list(PRESETS)}") from None
