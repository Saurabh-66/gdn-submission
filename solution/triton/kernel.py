"""
GDN Decode Kernel – MLSys 2026 FlashInfer Contest
Definition: gdn_decode_qk4_v8_d128_k_last

Exact signature from definition JSON:
    run(q, k, v, state, A_log, a, dt_bias, b, scale) -> (output, new_state)

Shapes (B=batch, T=1, HQ=4, HK=4, HV=8, K=128, V=128):
    q        : [B, 1, 4,  128]  bf16
    k        : [B, 1, 4,  128]  bf16
    v        : [B, 1, 8,  128]  bf16
    state    : [B, 8,  128, 128]  fp32  layout [B, H, V, K]  (optional)
    A_log    : [8]               fp32
    a        : [B, 1, 8]         bf16
    dt_bias  : [8]               fp32
    b        : [B, 1, 8]         bf16
    scale    : scalar            fp32

Gate computation (from reference):
    g    = exp(-exp(A_log) * softplus(a + dt_bias))   # [B, 1, HV]
    beta = sigmoid(b)                                  # [B, 1, HV]

Delta rule (state stored as [V,K], internally computed as [K,V]):
    state_KV  = state_VK.T          (per batch, per head)
    old_state = g * state_KV
    old_v     = k @ old_state       # [K] @ [K,V] -> [V]
    new_v     = beta*v + (1-beta)*old_v
    state_KV  = old_state - k^T @ old_v + k^T @ new_v
    output    = scale * q @ state_KV

Optimizations for B200 (sm_100a):
    - tl.dot for all matrix ops (tensor core utilization)
    - fp32 throughout for correctness (atol=rtol=0.01)
    - Grid (B, HV): each CTA owns one (batch, v-head) pair
    - Single tile covers full 128x128 state
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gdn_decode_kernel(
    q_ptr,      # [B, HQ, K]  bf16  (squeezed T=1)
    k_ptr,      # [B, HV, K]  bf16  (after repeat_interleave)
    v_ptr,      # [B, HV, V]  bf16
    state_ptr,  # [B, HV, V, K]  fp32  k-last layout
    g_ptr,      # [B, HV]  fp32  gate
    beta_ptr,   # [B, HV]  fp32  update scale
    scale,      # scalar fp32
    out_ptr,    # [B, HV, V]  fp32 (written as bf16 after)
    new_state_ptr,  # [B, HV, V, K]  fp32
    # strides  q[B, HV, K]
    sq_b, sq_h, sq_k,
    # strides  k[B, HV, K]
    sk_b, sk_h, sk_k,
    # strides  v[B, HV, V]
    sv_b, sv_h, sv_v,
    # strides  state[B, HV, V, K]  (k-last: V then K)
    ss_b, ss_h, ss_v, ss_k,
    # strides  out[B, HV, V]
    so_b, so_h, so_v,
    # strides  new_state[B, HV, V, K]
    sn_b, sn_h, sn_v, sn_k,
    # sizes
    B:   tl.constexpr,
    HV:  tl.constexpr,
    K:   tl.constexpr,   # key/query head dim = 128
    V:   tl.constexpr,   # value head dim = 128
    BK:  tl.constexpr,   # next_power_of_2(K) = 128
    BV:  tl.constexpr,   # next_power_of_2(V) = 128
):
    """Grid: (B, HV)"""
    b  = tl.program_id(0)
    hv = tl.program_id(1)

    rk = tl.arange(0, BK)   # [BK]
    rv = tl.arange(0, BV)   # [BV]
    mk = rk < K
    mv = rv < V
    # mask for [BK, BV] state tile
    ms = mk[:, None] & mv[None, :]

    # ── load q, k, v as fp32 ────────────────────────────────────────────
    q = tl.load(q_ptr + b*sq_b + hv*sq_h + rk*sq_k, mask=mk, other=0.).to(tl.float32)
    k = tl.load(k_ptr + b*sk_b + hv*sk_h + rk*sk_k, mask=mk, other=0.).to(tl.float32)
    v = tl.load(v_ptr + b*sv_b + hv*sv_h + rv*sv_v, mask=mv, other=0.).to(tl.float32)

    # ── load g, beta ────────────────────────────────────────────────────
    g    = tl.load(g_ptr    + b*HV + hv).to(tl.float32)
    beta = tl.load(beta_ptr + b*HV + hv).to(tl.float32)

    # ── load state in k-last layout [V, K] → we work with it as [K, V] ─
    # state_ptr is [B, HV, V, K].  We load it transposed: we want S[k, v].
    # Stride along K is ss_k (innermost), along V is ss_v.
    # We load S_KV[rk, rv] = state[b, hv, rv, rk]
    # offset = b*ss_b + hv*ss_h + rv[None,:]*ss_v + rk[:,None]*ss_k   -> [BK, BV]
    s_base = state_ptr + b*ss_b + hv*ss_h
    # Load transposed: rows=K, cols=V
    s_off_KV = rk[:, None]*ss_k + rv[None, :]*ss_v   # [BK, BV]
    S_KV = tl.load(s_base + s_off_KV, mask=ms, other=0.).to(tl.float32)

    # ── delta rule ───────────────────────────────────────────────────────
    # old_state = g * S_KV            [K, V]
    old_state = g * S_KV             # [BK, BV]

    # old_v = k @ old_state           [V]  =  k[1,K] @ old_state[K,V]
    k_row   = tl.reshape(k, (1, BK))
    old_v   = tl.reshape(tl.dot(k_row, old_state, allow_tf32=False), (BV,))   # [BV]

    # new_v = beta*v + (1-beta)*old_v
    new_v = beta * v + (1.0 - beta) * old_v                                   # [BV]

    # state_remove = k^T @ old_v      [K, V]
    k_col        = tl.reshape(k,     (BK, 1))
    old_v_row    = tl.reshape(old_v, (1,  BV))
    state_remove = tl.dot(k_col, old_v_row, allow_tf32=False)                 # [BK, BV]

    # state_update = k^T @ new_v      [K, V]
    new_v_row    = tl.reshape(new_v, (1,  BV))
    state_update = tl.dot(k_col, new_v_row, allow_tf32=False)                 # [BK, BV]

    # new S_KV
    new_S_KV = old_state - state_remove + state_update                        # [BK, BV]

    # ── output = scale * q @ new_S_KV  [V] ─────────────────────────────
    o = scale * tl.reshape(tl.dot(k_row, new_S_KV, allow_tf32=False), (BV,))
    # (reuse k_row shape; actually need q_row)
    q_row = tl.reshape(q, (1, BK))
    o     = scale * tl.reshape(tl.dot(q_row, new_S_KV, allow_tf32=False), (BV,))

    # ── store output ─────────────────────────────────────────────────────
    tl.store(out_ptr + b*so_b + hv*so_h + rv*so_v, o.to(tl.bfloat16), mask=mv)

    # ── store new_state in k-last [V, K] layout ──────────────────────────
    # new_state[b, hv, rv, rk] = new_S_KV[rk, rv]
    # same s_off_KV offset works: rows=K cols=V but stored into [V,K] base
    ns_base = new_state_ptr + b*sn_b + hv*sn_h
    ns_off  = rk[:, None]*sn_k + rv[None, :]*sn_v   # [BK, BV] -> store S_KV into [V,K]
    tl.store(ns_base + ns_off, new_S_KV, mask=ms)


def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """
    GDN decode – DPS entry point.

    Inputs:
        q        [B, 1, 4,  128]  bf16
        k        [B, 1, 4,  128]  bf16
        v        [B, 1, 8,  128]  bf16
        state    [B, 8,  128, 128]  fp32  [B,H,V,K]  (may be None)
        A_log    [8]               fp32
        a        [B, 1, 8]         bf16
        dt_bias  [8]               fp32
        b        [B, 1, 8]         bf16
        scale    scalar            fp32
    Outputs (pre-allocated, DPS):
        output    [B, 1, 8,  128]  bf16
        new_state [B, 8,  128, 128]  fp32
    """
    B, _, HQ, K = q.shape
    _, _, HV, V = v.shape
    dev = q.device

    # ── gate and beta on GPU (cheap elementwise) ─────────────────────────
    a_f    = a.float().squeeze(1)        # [B, HV]
    b_f    = b.float().squeeze(1)        # [B, HV]
    x      = a_f + dt_bias.float()      # [B, HV]  broadcast
    g_full = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))   # [B, HV]
    beta_f = torch.sigmoid(b_f)         # [B, HV]

    if scale is None or (isinstance(scale, float) and scale == 0.0):
        scale = 1.0 / math.sqrt(K)
    if isinstance(scale, torch.Tensor):
        scale = scale.item()

    # ── handle optional state ─────────────────────────────────────────────
    if state is None:
        state = torch.zeros(B, HV, V, K, dtype=torch.float32, device=dev)

    # ── repeat_interleave q,k to HV heads (GVA expansion) ────────────────
    GQA   = HV // HQ
    q_sq  = q.squeeze(1)                              # [B, HQ, K]
    k_sq  = k.squeeze(1)                              # [B, HQ, K]
    v_sq  = v.squeeze(1)                              # [B, HV, V]
    q_exp = q_sq.repeat_interleave(GQA, dim=1)        # [B, HV, K]
    k_exp = k_sq.repeat_interleave(GQA, dim=1)        # [B, HV, K]

    # ── contiguous ────────────────────────────────────────────────────────
    q_exp  = q_exp.contiguous()
    k_exp  = k_exp.contiguous()
    v_sq   = v_sq.contiguous()
    state  = state.contiguous()
    g_full = g_full.contiguous()
    beta_f = beta_f.contiguous()

    # Output buffer for kernel (stores [B, HV, V] then we unsqueeze)
    out_hv = torch.empty(B, HV, V, dtype=torch.bfloat16, device=dev)

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)

    _gdn_decode_kernel[(B, HV)](
        q_exp, k_exp, v_sq, state, g_full, beta_f, scale,
        out_hv, new_state,
        # q strides [B, HV, K]
        q_exp.stride(0), q_exp.stride(1), q_exp.stride(2),
        # k strides
        k_exp.stride(0), k_exp.stride(1), k_exp.stride(2),
        # v strides [B, HV, V]
        v_sq.stride(0), v_sq.stride(1), v_sq.stride(2),
        # state strides [B, HV, V, K]
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        # out strides [B, HV, V]
        out_hv.stride(0), out_hv.stride(1), out_hv.stride(2),
        # new_state strides [B, HV, V, K]
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        B=B, HV=HV, K=K, V=V, BK=BK, BV=BV,
        num_warps=4,
        num_stages=1,
    )

    # Unsqueeze T=1 dim back: [B, HV, V] -> [B, 1, HV, V]
    output.copy_(out_hv.unsqueeze(1))
    return output, new_state
