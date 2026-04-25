"""
GDN Prefill Kernel – MLSys 2026 FlashInfer Contest
Definition: gdn_prefill_qk4_v8_d128_k_last

Signature (from definition JSON):
    kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state)

Shapes:
    q          [total_seq_len, 4,  128]  bf16   (packed, variable-length seqs)
    k          [total_seq_len, 4,  128]  bf16
    v          [total_seq_len, 8,  128]  bf16
    state      [num_seqs, 8, 128, 128]   fp32   [N,H,V,K] k-last  (optional)
    A_log      [8]                       fp32
    a          [total_seq_len, 8]        bf16
    dt_bias    [8]                       fp32
    b          [total_seq_len, 8]        bf16
    cu_seqlens [num_seqs+1]              int64
    scale      scalar                    fp32
    output     [total_seq_len, 8, 128]   bf16   pre-allocated (DPS)
    new_state  [num_seqs, 8, 128, 128]   fp32   pre-allocated (DPS)

Algorithm (per token t, per head hv):
    g    = exp(-exp(A_log[hv]) * softplus(a[t,hv] + dt_bias[hv]))
    beta = sigmoid(b[t,hv])
    -- working in K,V space (state stored as [V,K] on disk, transposed to [K,V]) --
    old_state = g * S_KV
    old_v     = k[t] @ old_state        # [V]
    new_v     = beta*v[t] + (1-beta)*old_v
    S_KV      = old_state + k[t]^T ⊗ (new_v - old_v)
    out[t]    = scale * q[t] @ S_KV    # [V]

Kernel strategy:
    Grid: (num_seqs * HV,)
    Each CTA handles one (sequence, head) pair sequentially over T_seq tokens.
    This is optimal for prefill because:
    - State [K,V] = [128,128] fp32 = 64KB fits in L1/registers
    - Sequential scan avoids all synchronization
    - Matvec and outer product use elementwise ops (tl.dot needs K>=16 but
      we use tl.sum tricks to stay correct for all shapes)
    - For long sequences (T>>1) the compute is fully hidden behind memory BW

    For the inner loop over T_seq tokens we use the same elementwise approach
    as the decode kernel (no tl.dot) which avoids the K>=16 constraint and
    works correctly on all Triton versions.
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gdn_prefill_kernel(
    # inputs (pre-computed, fp32, expanded to HV heads)
    q_ptr,      # [total_T, HV, K]  fp32
    k_ptr,      # [total_T, HV, K]  fp32
    v_ptr,      # [total_T, HV, V]  fp32
    g_ptr,      # [total_T, HV]     fp32  gate
    beta_ptr,   # [total_T, HV]     fp32  beta
    # state in [N, HV, V, K] fp32  k-last on disk
    state_ptr,
    # output [total_T, HV, V] bf16
    out_ptr,
    # new_state [N, HV, V, K] fp32
    new_state_ptr,
    # cu_seqlens [N+1] int64
    cu_seqlens_ptr,
    scale,
    # strides q/k [total_T, HV, K]
    sq_t, sq_h, sq_k,
    sk_t, sk_h, sk_k,
    # strides v [total_T, HV, V]
    sv_t, sv_h, sv_v,
    # strides g/beta [total_T, HV]
    sg_t, sg_h,
    sbeta_t, sbeta_h,
    # strides state/new_state [N, HV, V, K]
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
    """
    Grid: (num_seqs * HV,)
    pid = seq_idx * HV + hv_idx
    """
    pid    = tl.program_id(0)
    seq_idx = pid // HV
    hv      = pid  % HV

    # ── sequence bounds ───────────────────────────────────────────────────
    t0 = tl.load(cu_seqlens_ptr + seq_idx).to(tl.int32)
    t1 = tl.load(cu_seqlens_ptr + seq_idx + 1).to(tl.int32)
    T_seq = t1 - t0

    rk = tl.arange(0, BK)
    rv = tl.arange(0, BV)
    mk = rk < K
    mv = rv < V
    ms = mk[:, None] & mv[None, :]   # [BK, BV]

    # ── load initial state [V,K] on disk -> [BK,BV] in registers ─────────
    s_base  = state_ptr + seq_idx*ss_n + hv*ss_h
    s_off   = rk[:, None]*ss_k + rv[None, :]*ss_v   # [BK,BV]: S_KV[k,v]=state[v,k]
    S_KV    = tl.load(s_base + s_off, mask=ms, other=0.).to(tl.float32)

    # ── sequential scan over tokens ───────────────────────────────────────
    for i in tl.range(0, T_seq, 1):
        t = t0 + i

        q    = tl.load(q_ptr    + t*sq_t + hv*sq_h + rk*sq_k,    mask=mk, other=0.).to(tl.float32)
        k    = tl.load(k_ptr    + t*sk_t + hv*sk_h + rk*sk_k,    mask=mk, other=0.).to(tl.float32)
        v    = tl.load(v_ptr    + t*sv_t + hv*sv_h + rv*sv_v,    mask=mv, other=0.).to(tl.float32)
        g    = tl.load(g_ptr    + t*sg_t + hv*sg_h).to(tl.float32)
        beta = tl.load(beta_ptr + t*sbeta_t + hv*sbeta_h).to(tl.float32)

        # old_state = g * S_KV
        old_state = g * S_KV                                    # [BK, BV]

        # old_v = k @ old_state  [BV]
        old_v = tl.sum(k[:, None] * old_state, axis=0)         # [BV]

        # new_v = beta*v + (1-beta)*old_v
        new_v = beta * v + (1.0 - beta) * old_v                # [BV]

        # delta_v = new_v - old_v
        delta_v = new_v - old_v                                 # [BV]

        # S_KV = old_state + k^T ⊗ delta_v
        S_KV = old_state + k[:, None] * delta_v[None, :]       # [BK, BV]

        # output = scale * q @ S_KV  [BV]
        o = scale * tl.sum(q[:, None] * S_KV, axis=0)          # [BV]

        tl.store(out_ptr + t*so_t + hv*so_h + rv*so_v, o.to(tl.bfloat16), mask=mv)

    # ── store new state [V,K] on disk ─────────────────────────────────────
    ns_base = new_state_ptr + seq_idx*sn_n + hv*sn_h
    ns_off  = rk[:, None]*sn_k + rv[None, :]*sn_v
    tl.store(ns_base + ns_off, S_KV, mask=ms)


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

    # ── compute g and beta for all tokens ────────────────────────────────
    a_f      = a.float()                                              # [total_T, HV]
    b_f      = b.float()                                              # [total_T, HV]
    x        = a_f + dt_bias.float()                                 # [total_T, HV]
    g_all    = torch.exp(-torch.exp(A_log.float()) * F.softplus(x)) # [total_T, HV]
    beta_all = torch.sigmoid(b_f)                                    # [total_T, HV]

    # ── GQA expansion: q,k from HQ heads -> HV heads ─────────────────────
    q_exp = q.float().repeat_interleave(GQA, dim=1).contiguous()    # [total_T, HV, K]
    k_exp = k.float().repeat_interleave(GQA, dim=1).contiguous()    # [total_T, HV, K]
    v_f   = v.float().contiguous()                                   # [total_T, HV, V]

    # ── handle optional state ─────────────────────────────────────────────
    if state is None:
        state = torch.zeros(num_seqs, HV, V, K, dtype=torch.float32, device=dev)
    state = state.contiguous()

    g_all    = g_all.contiguous()
    beta_all = beta_all.contiguous()
    cu_seqlens = cu_seqlens.contiguous()

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)

    # Grid: one CTA per (seq, head)
    grid = (num_seqs * HV,)

    _gdn_prefill_kernel[grid](
        q_exp, k_exp, v_f, g_all, beta_all,
        state, output, new_state, cu_seqlens, scale,
        # strides q [total_T, HV, K]
        q_exp.stride(0), q_exp.stride(1), q_exp.stride(2),
        # strides k
        k_exp.stride(0), k_exp.stride(1), k_exp.stride(2),
        # strides v [total_T, HV, V]
        v_f.stride(0), v_f.stride(1), v_f.stride(2),
        # strides g, beta [total_T, HV]
        g_all.stride(0), g_all.stride(1),
        beta_all.stride(0), beta_all.stride(1),
        # strides state [N, HV, V, K]
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        # strides new_state [N, HV, V, K]
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        # strides output [total_T, HV, V]
        output.stride(0), output.stride(1), output.stride(2),
        HV=HV, K=K, V=V, BK=BK, BV=BV,
        num_warps=4,
        num_stages=2,
    )

    return output, new_state
