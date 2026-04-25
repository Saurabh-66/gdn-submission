"""
GDN Prefill Kernel – MLSys 2026 FlashInfer Contest
Definition: gdn_prefill_qk4_v8_d128_k_last

The prefill recurrence is inherently sequential per (seq, head) — each token
depends on the previous state. So we parallelize across (seq, head) pairs
using a Triton kernel that holds the state in registers and scans tokens,
but we avoid a dynamic tl.range loop (which hangs on some Triton versions).

Fix: use a PyTorch-level loop over sequences, with a Triton kernel that
processes all tokens for one (seq, head) pair. The Triton kernel receives
T_seq as a constexpr via a Python-side dispatch over bucketed sequence lengths.

Actually simplest correct fix: pure batched PyTorch using einsum/matmul.
The reference is O(T * H * K * V) = O(T * 8 * 128 * 128). For the workload
sizes here this is fast enough AND correct. We beat the reference because the
reference has Python loops over seq_idx and token i — we vectorize over heads.

For maximum speedup we use a Triton kernel with a FIXED upper-bound loop
(T_MAX constexpr) and mask out-of-bounds tokens. We bucket by sequence length.
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gdn_prefill_seq_kernel(
    q_ptr,      # [T_seq, K]  fp32
    k_ptr,      # [T_seq, K]  fp32
    v_ptr,      # [T_seq, V]  fp32
    g_ptr,      # [T_seq]     fp32
    beta_ptr,   # [T_seq]     fp32
    s_ptr,      # [V, K]      fp32  initial state (k-last on disk)
    out_ptr,    # [T_seq, V]  bf16
    ns_ptr,     # [V, K]      fp32  new state (k-last on disk)
    scale,
    # strides
    sq_t, sq_k,
    sk_t, sk_k,
    sv_t, sv_v,
    ss_v, ss_k,
    so_t, so_v,
    sn_v, sn_k,
    # sizes
    T_seq,      # runtime int32 — used only for masking, loop bound is T_MAX
    K:  tl.constexpr,
    V:  tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    T_MAX: tl.constexpr,   # fixed upper bound for the loop (next_power_of_2 bucket)
):
    """Grid: (1,) — one CTA per (seq, head). Scans all T_MAX slots, masks extras."""
    rk = tl.arange(0, BK)
    rv = tl.arange(0, BV)
    mk = rk < K
    mv = rv < V
    ms = mk[:, None] & mv[None, :]

    # Load state [V,K] on disk -> S_KV [BK,BV] in registers
    s_off = rk[:, None] * ss_k + rv[None, :] * ss_v
    S_KV  = tl.load(s_ptr + s_off, mask=ms, other=0.).to(tl.float32)

    for i in tl.static_range(T_MAX):
        if i < T_seq:
            q    = tl.load(q_ptr    + i*sq_t + rk*sq_k,    mask=mk, other=0.).to(tl.float32)
            k    = tl.load(k_ptr    + i*sk_t + rk*sk_k,    mask=mk, other=0.).to(tl.float32)
            v    = tl.load(v_ptr    + i*sv_t + rv*sv_v,    mask=mv, other=0.).to(tl.float32)
            g    = tl.load(g_ptr    + i).to(tl.float32)
            beta = tl.load(beta_ptr + i).to(tl.float32)

            old_state = g * S_KV
            old_v     = tl.sum(k[:, None] * old_state, axis=0)      # [BV]
            new_v     = beta * v + (1.0 - beta) * old_v             # [BV]
            delta_v   = new_v - old_v                                # [BV]
            S_KV      = old_state + k[:, None] * delta_v[None, :]   # [BK,BV]
            o         = scale * tl.sum(q[:, None] * S_KV, axis=0)   # [BV]

            tl.store(out_ptr + i*so_t + rv*so_v, o.to(tl.bfloat16), mask=mv)

    # Store new state [K,V] -> [V,K] on disk
    ns_off = rk[:, None] * sn_k + rv[None, :] * sn_v
    tl.store(ns_ptr + ns_off, S_KV, mask=ms)


# Bucket sequence lengths to avoid recompiling for every unique T_seq
def _bucket(T):
    """Return next power of 2, capped at 4096."""
    b = 1
    while b < T:
        b <<= 1
    return min(b, 4096)


def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale,
           output, new_state):
    """
    GDN prefill – DPS entry point.

    q          [total_T, 4,  128]  bf16
    k          [total_T, 4,  128]  bf16
    v          [total_T, 8,  128]  bf16
    state      [N, 8, 128, 128]    fp32  [N,H,V,K]  (optional)
    A_log      [8]                 fp32
    a          [total_T, 8]        bf16
    dt_bias    [8]                 fp32
    b          [total_T, 8]        bf16
    cu_seqlens [N+1]               int64
    scale      scalar              fp32
    output     [total_T, 8, 128]   bf16  pre-allocated
    new_state  [N, 8, 128, 128]    fp32  pre-allocated
    """
    total_T, HQ, K = q.shape
    _,       HV, V = v.shape
    num_seqs = cu_seqlens.shape[0] - 1
    dev = q.device
    GQA = HV // HQ

    if scale is None or (isinstance(scale, float) and scale == 0.0):
        scale = 1.0 / math.sqrt(K)
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())

    # ── precompute g, beta for all tokens ────────────────────────────────
    a_f      = a.float()
    b_f      = b.float()
    x        = a_f + dt_bias.float()
    g_all    = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [total_T, HV]
    beta_all = torch.sigmoid(b_f)                                     # [total_T, HV]

    # ── GQA expansion ────────────────────────────────────────────────────
    q_exp = q.float().repeat_interleave(GQA, dim=1).contiguous()    # [total_T, HV, K]
    k_exp = k.float().repeat_interleave(GQA, dim=1).contiguous()    # [total_T, HV, K]
    v_f   = v.float().contiguous()                                   # [total_T, HV, V]

    if state is None:
        state = torch.zeros(num_seqs, HV, V, K, dtype=torch.float32, device=dev)
    state = state.contiguous()

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)

    # ── launch one Triton CTA per (seq, head) ────────────────────────────
    for seq_idx in range(num_seqs):
        t0    = int(cu_seqlens[seq_idx].item())
        t1    = int(cu_seqlens[seq_idx + 1].item())
        T_seq = t1 - t0
        if T_seq <= 0:
            new_state[seq_idx] = state[seq_idx]
            continue

        T_MAX = _bucket(T_seq)

        for hv in range(HV):
            q_sl   = q_exp[t0:t1, hv, :].contiguous()     # [T_seq, K]
            k_sl   = k_exp[t0:t1, hv, :].contiguous()     # [T_seq, K]
            v_sl   = v_f  [t0:t1, hv, :].contiguous()     # [T_seq, V]
            g_sl   = g_all  [t0:t1, hv].contiguous()      # [T_seq]
            beta_sl= beta_all[t0:t1, hv].contiguous()     # [T_seq]
            s_sl   = state[seq_idx, hv].contiguous()       # [V, K]
            o_sl   = output[t0:t1, hv, :].contiguous()    # [T_seq, V]
            ns_sl  = new_state[seq_idx, hv]                # [V, K]

            _gdn_prefill_seq_kernel[(1,)](
                q_sl, k_sl, v_sl, g_sl, beta_sl,
                s_sl, o_sl, ns_sl, scale,
                q_sl.stride(0), q_sl.stride(1),
                k_sl.stride(0), k_sl.stride(1),
                v_sl.stride(0), v_sl.stride(1),
                s_sl.stride(0), s_sl.stride(1),
                o_sl.stride(0), o_sl.stride(1),
                ns_sl.stride(0), ns_sl.stride(1),
                T_seq=T_seq,
                K=K, V=V, BK=BK, BV=BV,
                T_MAX=T_MAX,
                num_warps=4,
                num_stages=1,
            )
            output[t0:t1, hv, :] = o_sl

    return output, new_state
