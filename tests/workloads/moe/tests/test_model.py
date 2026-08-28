"""Full-model tests: all three presets forward/backward, learning, caches, Muon."""

import torch
import pytest

from latentmoe import LatentMoEModel, get_preset
from latentmoe.optim import MuonClip
from latentmoe.optim.muon import newton_schulz

torch.manual_seed(0)

MINI = dict(vocab_size=256, dim=64, n_layers=4, n_heads=4, head_dim=16,
            kv_latent_dim=32, q_latent_dim=32, rope_dim=16,
            csa_block=4, csa_topk=4, hca_block=8, local_window=16,
            indexer_heads=2, indexer_dim=8, out_groups=2, out_group_dim=16,
            kda_head_dim=16, kda_chunk=4,
            n_routed_experts=8, top_k=2, moe_latent_dim=32, expert_hidden=16,
            shared_expert_hidden=32, dense_ffn_hidden=64, mhc_streams=2,
            max_seq_len=128)


def _model(arch, **overrides):
    cfg = get_preset(arch, **{**MINI, **overrides})
    return LatentMoEModel(cfg), cfg


@pytest.mark.parametrize("arch", ["v4", "k3", "hybrid"])
def test_forward_backward_all_archs(arch):
    model, cfg = _model(arch)
    x = torch.randint(4, cfg.vocab_size, (2, 32))
    y = torch.randint(4, cfg.vocab_size, (2, 32))
    out = model(x, targets=y)
    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == (2, 32, cfg.vocab_size)
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    for g in grads:
        assert torch.isfinite(g).all()


@pytest.mark.parametrize("arch", ["v4", "k3"])
def test_model_learns(arch):
    torch.manual_seed(0)
    model, cfg = _model(arch, use_mtp=False)
    opt = MuonClip.from_model(model, lr=2e-3)
    x = torch.randint(4, cfg.vocab_size, (4, 32))
    y = torch.roll(x, -1, dims=1)  # trivially learnable shifted copy
    first = None
    model.train()
    for step in range(30):
        out = model(x, targets=y, return_logits=False)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
        model.balance_step()
        if first is None:
            first = float(out["ce_loss"].detach())
    last = float(out["ce_loss"].detach())
    assert last < first * 0.7, (first, last)


@pytest.mark.parametrize("arch", ["v4", "k3", "hybrid"])
def test_model_cache_matches_full(arch):
    torch.manual_seed(0)
    model, cfg = _model(arch)
    model.eval()
    T = 33
    x = torch.randint(4, cfg.vocab_size, (1, T))
    with torch.no_grad():
        full = model(x)["logits"]
        caches = model.new_caches()
        outs = [model(x[:, :17], caches=caches)["logits"]]
        for t in range(17, T):
            outs.append(model(x[:, t:t + 1], caches=caches)["logits"])
        inc = torch.cat(outs, dim=1)
    assert torch.allclose(full, inc, atol=3e-4), (full - inc).abs().max()


def test_newton_schulz_orthogonalizes():
    g = torch.randn(32, 96)
    o = newton_schulz(g)
    eye = o @ o.t()
    assert torch.allclose(eye, torch.eye(32), atol=0.15), \
        (eye - torch.eye(32)).abs().max()


def test_muon_per_head_and_qk_clip():
    model, cfg = _model("k3")
    opt = MuonClip.from_model(model, lr=1e-3)
    x = torch.randint(4, cfg.vocab_size, (2, 16))
    y = torch.randint(4, cfg.vocab_size, (2, 16))
    model.train()
    model.set_track_qk(True)
    out = model(x, targets=y, return_logits=False)
    out["loss"].backward()
    opt.step()
    # force a clip: pretend a head saturated
    for m in model.modules():
        if hasattr(m, "max_qk_logit"):
            m.max_qk_logit.fill_(500.0)
    before = [m.w_uq.weight.clone() for m in model.modules() if hasattr(m, "w_uq")]
    opt.qk_clip(model, tau=100.0)
    after = [m.w_uq.weight for m in model.modules() if hasattr(m, "w_uq")]
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert changed


def test_mtp_speculative_pathway():
    model, cfg = _model("k3", mtp_fuse_layers=True)
    model.eval()
    x = torch.randint(4, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        out = model(x)
        h = out["hidden_prenorm"][:, -1:]
        feats = [f[:, -1:] for f in out["mtp_feats"]]
        nxt = out["logits"][:, -1].argmax(-1, keepdim=True)
        d = model.mtp(h, model.embed(nxt), feats=feats)
        logits = model.head(d)
    assert logits.shape == (1, 1, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_qat_model_runs():
    model, cfg = _model("k3", qat_experts=True)
    x = torch.randint(4, cfg.vocab_size, (2, 16))
    y = torch.randint(4, cfg.vocab_size, (2, 16))
    out = model(x, targets=y, return_logits=False)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])


# --------------------------------------------------------------------- #
class _MuonToy(torch.nn.Module):
    """Mix of Muon param kinds: two same-shape 2D (batch together), one
    distinct 2D that shape-matches the per-head slices of a tagged q
    projection (cross-kind batching), and a 3D expert stack."""

    def __init__(self):
        super().__init__()
        nn = torch.nn
        self.a = nn.Linear(32, 32, bias=False)
        self.b = nn.Linear(32, 32, bias=False)
        self.c = nn.Linear(32, 16, bias=False)     # [16,32] == q's head slices
        self.q = nn.Linear(32, 64, bias=False)     # [64,32] -> 4 x [16,32]
        self.q.weight._muon_heads = 4
        self.w3 = torch.nn.Parameter(torch.randn(4, 16, 8) * 0.02)


def test_muon_batched_ns_matches_per_param():
    def build():
        torch.manual_seed(3)
        m = _MuonToy()
        torch.manual_seed(7)
        for p in m.parameters():
            p.grad = torch.randn_like(p)
        return m

    m_ref, m_bat = build(), build()
    MuonClip.from_model(m_ref, lr=1e-2, ns_batched=False).step()
    MuonClip.from_model(m_bat, lr=1e-2, ns_batched=True).step()
    for (n, p1), (_, p2) in zip(m_ref.named_parameters(), m_bat.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), (n, (p1 - p2).abs().max())


# --------------------------------------------------------------------- #
@pytest.mark.parametrize("arch", ["k3", "v4"])
def test_exact_speculative_matches_greedy(arch):
    """Exact speculative decoding (fork_caches snapshot + rewind/replay on
    rejection) must produce the IDENTICAL token sequence as plain greedy
    decoding. Untrained model => most drafts reject => the rewind path is
    exercised hard, across all cache families (KDA state / MLA latent for
    k3, CSA compressed+tail for v4)."""
    torch.manual_seed(0)
    cfg = get_preset(arch, **MINI)
    model = LatentMoEModel(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 24))
    gen_n = 12

    @torch.no_grad()
    def greedy():
        caches = model.new_caches()
        out = model(prompt, caches=caches, return_logits=True)
        step = out["logits"][:, -1].argmax(-1, keepdim=True)
        toks = [int(step)]
        while len(toks) < gen_n:
            out = model(step, caches=caches, return_logits=True)
            step = out["logits"][:, -1].argmax(-1, keepdim=True)
            toks.append(int(step))
        return toks

    @torch.no_grad()
    def spec_exact():
        caches = model.new_caches()
        out = model(prompt, caches=caches, return_logits=True)
        step = out["logits"][:, -1].argmax(-1, keepdim=True)
        toks = [int(step)]

        def draft_from(o, nxt):
            feats = ([f[:, -1:] for f in o["mtp_feats"]]
                     if cfg.mtp_fuse_layers else None)
            d_h = model.mtp(o["hidden_prenorm"][:, -1:], model.embed(nxt),
                            feats=feats)
            return model.head(d_h)[:, -1].argmax(-1, keepdim=True)

        draft = draft_from(out, step)
        while len(toks) < gen_n:
            snap = model.fork_caches(caches)
            out = model(torch.cat([step, draft], dim=1), caches=caches,
                        return_logits=True)
            verify = out["logits"][:, 0].argmax(-1, keepdim=True)
            if int(verify) == int(draft):                 # accepted: +2
                nxt = out["logits"][:, 1].argmax(-1, keepdim=True)
                toks.append(int(draft))
                if len(toks) < gen_n:
                    toks.append(int(nxt))
                step = nxt
                draft = draft_from(out, nxt)
                continue
            for c, s in zip(caches, snap):                # rejected: rewind
                c.clear()
                c.update(s)
            out = model(step, caches=caches, return_logits=True)
            toks.append(int(verify))
            step = verify
            draft = draft_from(out, step)
        return toks

    assert greedy() == spec_exact()


def test_gram_chunked_matches_direct():
    """K-chunked Gram accumulation (deterministic-at-any-shape path for
    Muon's Newton-Schulz, HANDOFF UPDATE 24) must match the monolithic
    X @ X^T product to accumulation-rounding tolerance, 2D and batched,
    including a non-multiple-of-chunk K."""
    from latentmoe.optim.muon import _GRAM_K_CHUNK, _det_mm, _gram
    g = torch.Generator().manual_seed(7)
    for shape in ((1024, 3072), (8, _GRAM_K_CHUNK + 1), (2, 64, 5000)):
        x = torch.randn(*shape, generator=g)
        # chunked vs monolithic differ only by fp32 accumulation order:
        # tiny absolute (~1e-4 on O(1e3) Gram entries), but near-zero
        # off-diagonals need an absolute band, not a relative one
        torch.testing.assert_close(_gram(x), x @ x.transpose(-2, -1),
                                   rtol=1e-5, atol=2e-3)
    # _det_mm general paths: N-chunk only (bit-free column slabs),
    # K-and-N chunk, and batched N-chunk (the B @ X wide-output case)
    for pshape, qshape in (((64, 1024), (1024, 3000)),
                           ((8, 2100), (2100, 2100)),
                           ((2, 16, 512), (2, 512, 3000))):
        p = torch.randn(*pshape, generator=g)
        q = torch.randn(*qshape, generator=g)
        torch.testing.assert_close(_det_mm(p, q), p @ q,
                                   rtol=1e-5, atol=2e-3)
    # small inputs take the untouched single-matmul path bit-exactly
    x = torch.randn(64, 128, generator=g)
    assert torch.equal(_gram(x), x @ x.transpose(-2, -1))
    # NS still orthogonalizes at the field-culprit geometry (1024x3072)
    x = torch.randn(1024, 3072, generator=g)
    o = newton_schulz(x)
    ident = o @ o.transpose(-2, -1)
    assert (ident - torch.eye(1024)).abs().max() < 0.35  # hybrid-NS quality
    # transpose identity NS(X^T) == NS(X)^T (canonical orientation relies
    # on it, UPDATE 26): exact in math, rounding-level in floats
    torch.testing.assert_close(newton_schulz(x.t().contiguous()).t(), o,
                               rtol=1e-4, atol=1e-4)
