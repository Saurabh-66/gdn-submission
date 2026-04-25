"""
GDN Prefill Kernel – MLSys 2026 FlashInfer Contest
Definition: gdn_prefill_qk4_v8_d128_k_last

Proper Triton kernel with runtime tl.range loop.
- tl.range(0, T_seq, 1) compiles to a normal PTX loop — no unrolling, no hang
- State S_KV [BK, BV] = [128, 128] lives in registers across the entire loop
- Zero HBM state traffic during the scan (only q/k/v/g/beta streamed per token)
- Grid: (num_seqs * HV,) — one CTA per (sequence, head)

Progress monitoring: kernel writes a counter to a shared tensor after each
sequence completes, which you can watch from a second terminal.
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gdn_prefill_kernel(
    # inputs [total_T, HV, *]  fp32 (pre-expanded)
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    # per-sequence offsets: t0[seq], t1[seq]
    t0_ptr,     # [num_seqs]  int32
    t1_ptr,     # [num_seqs]  int32
    # state [N, HV, V, K]  fp32  k-last on disk
    state_ptr,
    # output [total_T, HV, V]  bf16
    out_ptr,
    # new_state [N, HV, V, K]  fp32
    ns_ptr,
    # progress counter [1]  int32  atomic increment per completed CTA
    progress_ptr,
    scale,
    # strides q/k [total_T, HV, K]
    sq_t, sq_h, sq_k,
    sk_t, sk_h, sk_k,
    # strides v [total_T, HV, V]
    sv_t, sv_h, sv_v,
    # strides g/beta [total_T, HV]
    sg_t, sg_h,
    sbeta_t, sbeta_h,
    # strides state/ns [N, HV, V, K]
    ss_n, ss_h, ss_v, ss_k,
    sn_n, sn_h, sn_v, sn_k,
    # strides out [total_T, HV, V]
    so_t, so_h, so_v,
    # sizes
    HV:  tl.constexpr,
    K:   tl.constexpr,
    V:   tl.constexpr,
    BK:  tl.constexpr,
    BV:  tl.constexpr,
):
    """Grid: (num_seqs * HV,)"""
    pid     = tl.program_id(0)
    seq_idx = pid // HV
    hv      = pid  % HV

    t0    = tl.load(t0_ptr + seq_idx).to(tl.int32)
    t1    = tl.load(t1_ptr + seq_idx).to(tl.int32)
    T_seq = t1 - t0

    rk = tl.arange(0, BK)
    rv = tl.arange(0, BV)
    mk = rk < K
    mv = rv < V
    ms = mk[:, None] & mv[None, :]

    # Load initial state [V,K] on disk -> S_KV [BK,BV] in registers
    s_base = state_ptr + seq_idx * ss_n + hv * ss_h
    s_off  = rk[:, None] * ss_k + rv[None, :] * ss_v
    S_KV   = tl.load(s_base + s_off, mask=ms, other=0.).to(tl.float32)

    # Sequential scan — runtime loop, compiles to normal PTX loop
    for i in tl.range(0, T_seq, 1):
        t = t0 + i

        q    = tl.load(q_ptr    + t*sq_t + hv*sq_h + rk*sq_k, mask=mk, other=0.).to(tl.float32)
        k    = tl.load(k_ptr    + t*sk_t + hv*sk_h + rk*sk_k, mask=mk, other=0.).to(tl.float32)
        v    = tl.load(v_ptr    + t*sv_t + hv*sv_h + rv*sv_v, mask=mv, other=0.).to(tl.float32)
        g    = tl.load(g_ptr    + t*sg_t + hv*sg_h).to(tl.float32)
        beta = tl.load(beta_ptr + t*sbeta_t + hv*sbeta_h).to(tl.float32)

        old_state = g * S_KV
        old_v     = tl.sum(k[:, None] * old_state, axis=0)
        new_v     = beta * v + (1.0 - beta) * old_v
        delta_v   = new_v - old_v
        S_KV      = old_state + k[:, None] * delta_v[None, :]
        o         = scale * tl.sum(q[:, None] * S_KV, axis=0)

        tl.store(out_ptr + t*so_t + hv*so_h + rv*so_v, o.to(tl.bfloat16), mask=mv)

    # Store new state
    ns_base = ns_ptr + seq_idx * sn_n + hv * sn_h
    ns_off  = rk[:, None] * sn_k + rv[None, :] * sn_v
    tl.store(ns_base + ns_off, S_KV, mask=ms)

    # Atomic progress increment (only once per seq, from head 0)
    if hv == 0:
        tl.atomic_add(progress_ptr, 1)


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

    # ── precompute g, beta ────────────────────────────────────────────────
    x        = a.float() + dt_bias.float()
    g_all    = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [total_T, HV]
    beta_all = torch.sigmoid(b.float())                               # [total_T, HV]

    # ── GQA expansion ────────────────────────────────────────────────────
    q_exp = q.float().repeat_interleave(GQA, dim=1).contiguous()     # [total_T, HV, K]
    k_exp = k.float().repeat_interleave(GQA, dim=1).contiguous()     # [total_T, HV, K]
    v_f   = v.float().contiguous()                                    # [total_T, HV, V]
    g_all    = g_all.contiguous()
    beta_all = beta_all.contiguous()

    if state is None:
        state = torch.zeros(num_seqs, HV, V, K, dtype=torch.float32, device=dev)
    state = state.contiguous()

    # ── build t0, t1 arrays ───────────────────────────────────────────────
    cu = cu_seqlens.to(torch.int32)
    t0_arr = cu[:-1].contiguous()   # [num_seqs]
    t1_arr = cu[1: ].contiguous()   # [num_seqs]

    # ── progress counter ──────────────────────────────────────────────────
    progress = torch.zeros(1, dtype=torch.int32, device=dev)

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)

    grid = (num_seqs * HV,)

    _gdn_prefill_kernel[grid](
        q_exp, k_exp, v_f, g_all, beta_all,
        t0_arr, t1_arr,
        state, output, new_state,
        progress,
        scale,
        # q strides [total_T, HV, K]
        q_exp.stride(0), q_exp.stride(1), q_exp.stride(2),
        # k strides
        k_exp.stride(0), k_exp.stride(1), k_exp.stride(2),
        # v strides [total_T, HV, V]
        v_f.stride(0), v_f.stride(1), v_f.stride(2),
        # g strides [total_T, HV]
        g_all.stride(0), g_all.stride(1),
        # beta strides
        beta_all.stride(0), beta_all.stride(1),
        # state strides [N, HV, V, K]
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        # new_state strides
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        # output strides [total_T, HV, V]
        output.stride(0), output.stride(1), output.stride(2),
        HV=HV, K=K, V=V, BK=BK, BV=BV,
        num_warps=4,
        num_stages=2,
    )

    return output, new_state
