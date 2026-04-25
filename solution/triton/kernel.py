"""
GDN Decode Kernel – MLSys 2026 FlashInfer Contest
Definition: gdn_decode_qk4_v8_d128_k_last

Fix: tl.dot requires K>=16. For matvec and outer products we use
elementwise ops instead:
  - matvec  S[K,V] @ k[K]  =>  tl.sum(S * k[:,None], axis=0)  [V]
  - outer   k[K] x d[V]    =>  k[:,None] * d[None,:]           [K,V]
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gdn_decode_kernel(
    q_ptr,          # [B, HV, K]  fp32 (after repeat_interleave, squeezed T)
    k_ptr,          # [B, HV, K]  fp32
    v_ptr,          # [B, HV, V]  fp32
    state_ptr,      # [B, HV, V, K]  fp32  k-last layout
    g_ptr,          # [B, HV]  fp32
    beta_ptr,       # [B, HV]  fp32
    scale,          # scalar fp32
    out_ptr,        # [B, HV, V]  bf16
    new_state_ptr,  # [B, HV, V, K]  fp32
    # strides q/k [B, HV, K]
    sq_b, sq_h, sq_k,
    sk_b, sk_h, sk_k,
    # strides v [B, HV, V]
    sv_b, sv_h, sv_v,
    # strides state/new_state [B, HV, V, K]
    ss_b, ss_h, ss_v, ss_k,
    sn_b, sn_h, sn_v, sn_k,
    # strides out [B, HV, V]
    so_b, so_h, so_v,
    # sizes
    B:   tl.constexpr,
    HV:  tl.constexpr,
    K:   tl.constexpr,
    V:   tl.constexpr,
    BK:  tl.constexpr,
    BV:  tl.constexpr,
):
    """Grid: (B, HV)"""
    b  = tl.program_id(0)
    hv = tl.program_id(1)

    rk = tl.arange(0, BK)   # [BK]
    rv = tl.arange(0, BV)   # [BV]
    mk = rk < K
    mv = rv < V
    ms = mk[:, None] & mv[None, :]   # [BK, BV]

    # ── load q, k, v ────────────────────────────────────────────────────
    q = tl.load(q_ptr + b*sq_b + hv*sq_h + rk*sq_k, mask=mk, other=0.).to(tl.float32)  # [BK]
    k = tl.load(k_ptr + b*sk_b + hv*sk_h + rk*sk_k, mask=mk, other=0.).to(tl.float32)  # [BK]
    v = tl.load(v_ptr + b*sv_b + hv*sv_h + rv*sv_v, mask=mv, other=0.).to(tl.float32)  # [BV]

    # ── load g, beta ─────────────────────────────────────────────────────
    g    = tl.load(g_ptr    + b*HV + hv).to(tl.float32)
    beta = tl.load(beta_ptr + b*HV + hv).to(tl.float32)

    # ── load state [V, K] stored on disk, work as S_KV [BK, BV] ─────────
    # state[b,hv,v,k] -> S_KV[k,v]: offset = v*ss_v + k*ss_k
    s_base  = state_ptr + b*ss_b + hv*ss_h
    s_off   = rk[:, None]*ss_k + rv[None, :]*ss_v   # [BK, BV]
    S_KV    = tl.load(s_base + s_off, mask=ms, other=0.).to(tl.float32)

    # ── old_state = g * S_KV ─────────────────────────────────────────────
    old_state = g * S_KV    # [BK, BV]

    # ── old_v = k @ old_state  [BV] ──────────────────────────────────────
    # matvec: old_v[v] = sum_k k[k] * old_state[k,v]
    # = tl.sum(k[:,None] * old_state, axis=0)
    old_v = tl.sum(k[:, None] * old_state, axis=0)   # [BV]

    # ── new_v = beta*v + (1-beta)*old_v ──────────────────────────────────
    new_v = beta * v + (1.0 - beta) * old_v          # [BV]

    # ── state_delta = k^T @ (new_v - old_v)  [BK, BV] ───────────────────
    # outer product: delta_v = new_v - old_v
    delta_v = new_v - old_v                           # [BV]
    # outer: k[:,None] * delta_v[None,:]
    state_delta = k[:, None] * delta_v[None, :]      # [BK, BV]

    # ── new_S_KV ─────────────────────────────────────────────────────────
    new_S_KV = old_state + state_delta               # [BK, BV]

    # ── output = scale * q @ new_S_KV  [BV] ─────────────────────────────
    # matvec: o[v] = sum_k q[k] * new_S_KV[k,v]
    o = scale * tl.sum(q[:, None] * new_S_KV, axis=0)   # [BV]

    # ── store output ─────────────────────────────────────────────────────
    tl.store(out_ptr + b*so_b + hv*so_h + rv*so_v, o.to(tl.bfloat16), mask=mv)

    # ── store new_state [V, K] on disk ───────────────────────────────────
    ns_base = new_state_ptr + b*sn_b + hv*sn_h
    ns_off  = rk[:, None]*sn_k + rv[None, :]*sn_v
    tl.store(ns_base + ns_off, new_S_KV, mask=ms)


def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """
    GDN decode – DPS entry point.
    q        [B, 1, 4,  128]  bf16
    k        [B, 1, 4,  128]  bf16
    v        [B, 1, 8,  128]  bf16
    state    [B, 8,  128, 128]  fp32  [B,H,V,K]
    A_log    [8]               fp32
    a        [B, 1, 8]         bf16
    dt_bias  [8]               fp32
    b        [B, 1, 8]         bf16
    scale    scalar            fp32
    output   [B, 1, 8,  128]  bf16   pre-allocated
    new_state[B, 8,  128, 128]  fp32  pre-allocated
    """
    B, _, HQ, K = q.shape
    _, _, HV, V = v.shape
    dev = q.device
    GQA = HV // HQ

    # ── compute g and beta ───────────────────────────────────────────────
    a_f    = a.float().squeeze(1)           # [B, HV]
    b_f    = b.float().squeeze(1)           # [B, HV]
    x      = a_f + dt_bias.float()         # [B, HV]
    g_full = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [B, HV]
    beta_f = torch.sigmoid(b_f)            # [B, HV]

    if scale is None or (isinstance(scale, float) and scale == 0.0):
        scale = 1.0 / math.sqrt(K)
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())

    # ── handle optional state ────────────────────────────────────────────
    if state is None:
        state = torch.zeros(B, HV, V, K, dtype=torch.float32, device=dev)

    # ── GQA expansion ────────────────────────────────────────────────────
    q_sq  = q.squeeze(1).float().repeat_interleave(GQA, dim=1)   # [B, HV, K]
    k_sq  = k.squeeze(1).float().repeat_interleave(GQA, dim=1)   # [B, HV, K]
    v_sq  = v.squeeze(1).float()                                   # [B, HV, V]

    q_sq   = q_sq.contiguous()
    k_sq   = k_sq.contiguous()
    v_sq   = v_sq.contiguous()
    state  = state.contiguous()
    g_full = g_full.contiguous()
    beta_f = beta_f.contiguous()

    out_hv = torch.empty(B, HV, V, dtype=torch.bfloat16, device=dev)

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)

    _gdn_decode_kernel[(B, HV)](
        q_sq, k_sq, v_sq, state, g_full, beta_f, scale,
        out_hv, new_state,
        q_sq.stride(0),  q_sq.stride(1),  q_sq.stride(2),
        k_sq.stride(0),  k_sq.stride(1),  k_sq.stride(2),
        v_sq.stride(0),  v_sq.stride(1),  v_sq.stride(2),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        out_hv.stride(0), out_hv.stride(1), out_hv.stride(2),
        B=B, HV=HV, K=K, V=V, BK=BK, BV=BV,
        num_warps=4,
        num_stages=1,
    )

    output.copy_(out_hv.unsqueeze(1))
    return output, new_state
