"""
GDN Prefill Kernel – MLSys 2026 FlashInfer Contest
Definition: gdn_prefill_qk4_v8_d128_k_last

Optimizations:
  1. Chunked: outer loop T/64 iters (not T iters), inner loop constexpr 64
  2. Multi-sequence: all seqs launched in one grid
  3. Triton: state held in registers across chunk loop

Two-kernel design:
  Kernel 1 (_chunk_state): Grid (num_seqs, HV)
    - Holds S_KV in registers across ALL chunks sequentially
    - Saves S_pre[c] before processing each chunk
    - Writes final_state

  Kernel 2 (_chunk_output): Grid (num_seqs * NC, HV)
    - Loads S_pre[c], re-runs the EXACT same recurrence as Kernel 1
      but only for CS=64 tokens (constexpr loop, no hang)
    - Emits output at each token

Key correctness principle: Kernel 2 runs IDENTICAL math to Kernel 1.
Both implement:
    old_state = g * S_KV
    old_v     = sum_k(k * old_state)       matvec
    new_v     = beta*v + (1-beta)*old_v
    S_KV      = old_state + k[:,None] * (new_v - old_v)[None,:]
    output    = scale * sum_k(q * S_KV)    matvec
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


CS = 64   # chunk size — must be small enough that static_range(CS) doesn't hang


@triton.jit
def _chunk_state_kernel(
    k_ptr, v_ptr, g_ptr, beta_ptr,
    t0_ptr, t1_ptr,
    init_s_ptr,   # [N, HV, V, K]  k-last on disk
    spre_ptr,     # [N, NC, HV, K, V]  working layout (K before V)
    final_s_ptr,  # [N, HV, V, K]  k-last on disk
    sk_t, sk_h, sk_k,
    sv_t, sv_h, sv_v,
    sg_t, sg_h,
    sbeta_t, sbeta_h,
    si_n, si_h, si_v, si_k,
    sf_n, sf_h, sf_v, sf_k,
    sp_n, sp_c, sp_h, sp_k, sp_v,
    HV:  tl.constexpr,
    K:   tl.constexpr,
    V:   tl.constexpr,
    BK:  tl.constexpr,
    BV:  tl.constexpr,
    CS:  tl.constexpr,
    NC:  tl.constexpr,
):
    """Grid: (num_seqs, HV)"""
    seq_idx = tl.program_id(0)
    hv      = tl.program_id(1)

    t0    = tl.load(t0_ptr + seq_idx).to(tl.int32)
    t1    = tl.load(t1_ptr + seq_idx).to(tl.int32)
    T_seq = t1 - t0

    rk = tl.arange(0, BK)
    rv = tl.arange(0, BV)
    mk = rk < K
    mv = rv < V
    ms = mk[:, None] & mv[None, :]

    # Load initial state: disk layout [V,K] -> registers [BK,BV]
    is_base = init_s_ptr + seq_idx*si_n + hv*si_h
    is_off  = rk[:, None]*si_k + rv[None, :]*si_v
    S_KV    = tl.load(is_base + is_off, mask=ms, other=0.).to(tl.float32)

    sp_base = spre_ptr + seq_idx*sp_n + hv*sp_h
    sp_off  = rk[:, None]*sp_k + rv[None, :]*sp_v

    for c in tl.range(0, NC, 1):
        # Save S_pre BEFORE processing this chunk
        tl.store(sp_base + c*sp_c + sp_off, S_KV, mask=ms)

        chunk_t0 = c * CS

        for i in tl.static_range(CS):
            t_local = chunk_t0 + i
            tg      = t0 + t_local

            if t_local < T_seq:
                k    = tl.load(k_ptr    + tg*sk_t + hv*sk_h + rk*sk_k, mask=mk, other=0.).to(tl.float32)
                v    = tl.load(v_ptr    + tg*sv_t + hv*sv_h + rv*sv_v, mask=mv, other=0.).to(tl.float32)
                g    = tl.load(g_ptr    + tg*sg_t + hv*sg_h).to(tl.float32)
                beta = tl.load(beta_ptr + tg*sbeta_t + hv*sbeta_h).to(tl.float32)

                # Exact reference recurrence
                old_state = g * S_KV
                old_v     = tl.sum(k[:, None] * old_state, axis=0)    # [BV]
                new_v     = beta * v + (1.0 - beta) * old_v           # [BV]
                S_KV      = old_state + k[:, None] * (new_v - old_v)[None, :]

    # Write final state: registers [BK,BV] -> disk [V,K]
    fs_base = final_s_ptr + seq_idx*sf_n + hv*sf_h
    fs_off  = rk[:, None]*sf_k + rv[None, :]*sf_v
    tl.store(fs_base + fs_off, S_KV, mask=ms)


@triton.jit
def _chunk_output_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
    t0_ptr, t1_ptr,
    spre_ptr,   # [N, NC, HV, K, V]
    out_ptr,    # [total_T, HV, V]  bf16
    scale,
    sq_t, sq_h, sq_k,
    sk_t, sk_h, sk_k,
    sv_t, sv_h, sv_v,
    sg_t, sg_h,
    sbeta_t, sbeta_h,
    sp_n, sp_c, sp_h, sp_k, sp_v,
    so_t, so_h, so_v,
    HV:  tl.constexpr,
    K:   tl.constexpr,
    V:   tl.constexpr,
    BK:  tl.constexpr,
    BV:  tl.constexpr,
    CS:  tl.constexpr,
    NC:  tl.constexpr,
):
    """Grid: (num_seqs * NC, HV)"""
    pid     = tl.program_id(0)
    seq_idx = pid // NC
    c       = pid  % NC
    hv      = tl.program_id(1)

    t0    = tl.load(t0_ptr + seq_idx).to(tl.int32)
    t1    = tl.load(t1_ptr + seq_idx).to(tl.int32)
    T_seq = t1 - t0

    chunk_t0 = c * CS

    rk = tl.arange(0, BK)
    rv = tl.arange(0, BV)
    mk = rk < K
    mv = rv < V
    ms = mk[:, None] & mv[None, :]

    # Load S_pre for this chunk (state BEFORE chunk c)
    sp_base = spre_ptr + seq_idx*sp_n + c*sp_c + hv*sp_h
    sp_off  = rk[:, None]*sp_k + rv[None, :]*sp_v
    S_KV    = tl.load(sp_base + sp_off, mask=ms, other=0.).to(tl.float32)

    # Re-run IDENTICAL recurrence to Kernel 1, but also emit output
    for i in tl.static_range(CS):
        t_local = chunk_t0 + i
        tg      = t0 + t_local

        if t_local < T_seq:
            q    = tl.load(q_ptr    + tg*sq_t + hv*sq_h + rk*sq_k, mask=mk, other=0.).to(tl.float32)
            k    = tl.load(k_ptr    + tg*sk_t + hv*sk_h + rk*sk_k, mask=mk, other=0.).to(tl.float32)
            v    = tl.load(v_ptr    + tg*sv_t + hv*sv_h + rv*sv_v, mask=mv, other=0.).to(tl.float32)
            g    = tl.load(g_ptr    + tg*sg_t + hv*sg_h).to(tl.float32)
            beta = tl.load(beta_ptr + tg*sbeta_t + hv*sbeta_h).to(tl.float32)

            # Exact same recurrence as Kernel 1
            old_state = g * S_KV
            old_v     = tl.sum(k[:, None] * old_state, axis=0)
            new_v     = beta * v + (1.0 - beta) * old_v
            S_KV      = old_state + k[:, None] * (new_v - old_v)[None, :]

            # Output uses NEW S_KV (after update), with scale
            o = scale * tl.sum(q[:, None] * S_KV, axis=0)   # [BV]
            tl.store(out_ptr + tg*so_t + hv*so_h + rv*so_v,
                     o.to(tl.bfloat16), mask=mv)


def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale,
           output, new_state):
    total_T, HQ, K = q.shape
    _,       HV, V = v.shape
    num_seqs = cu_seqlens.shape[0] - 1
    dev = q.device
    GQA = HV // HQ

    if scale is None or (isinstance(scale, float) and scale == 0.0):
        scale = 1.0 / math.sqrt(K)
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())

    # precompute g, beta
    x        = a.float() + dt_bias.float()
    g_all    = torch.exp(-torch.exp(A_log.float()) * F.softplus(x)).contiguous()
    beta_all = torch.sigmoid(b.float()).contiguous()

    # GQA expansion
    q_exp = q.float().repeat_interleave(GQA, dim=1).contiguous()
    k_exp = k.float().repeat_interleave(GQA, dim=1).contiguous()
    v_f   = v.float().contiguous()

    if state is None:
        state = torch.zeros(num_seqs, HV, V, K, dtype=torch.float32, device=dev)
    state = state.contiguous()

    cu     = cu_seqlens.to(torch.int32)
    t0_arr = cu[:-1].contiguous()
    t1_arr = cu[1: ].contiguous()

    max_T  = int((t1_arr - t0_arr).max().item())
    NC     = math.ceil(max_T / CS)

    # S_pre buffer: [N, NC, HV, K, V]  working layout
    S_pre = torch.empty(num_seqs, NC, HV, K, V, dtype=torch.float32, device=dev)

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)

    # Kernel 1: state propagation, saves S_pre
    _chunk_state_kernel[(num_seqs, HV)](
        k_exp, v_f, g_all, beta_all,
        t0_arr, t1_arr,
        state, S_pre, new_state,
        k_exp.stride(0), k_exp.stride(1), k_exp.stride(2),
        v_f.stride(0),   v_f.stride(1),   v_f.stride(2),
        g_all.stride(0),    g_all.stride(1),
        beta_all.stride(0), beta_all.stride(1),
        # init_state [N, HV, V, K]
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        # final_state [N, HV, V, K]
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        # S_pre [N, NC, HV, K, V]
        S_pre.stride(0), S_pre.stride(1), S_pre.stride(2),
        S_pre.stride(3), S_pre.stride(4),
        HV=HV, K=K, V=V, BK=BK, BV=BV, CS=CS, NC=NC,
        num_warps=4, num_stages=1,
    )

    # Kernel 2: output computation
    _chunk_output_kernel[(num_seqs * NC, HV)](
        q_exp, k_exp, v_f, g_all, beta_all,
        t0_arr, t1_arr,
        S_pre, output, scale,
        q_exp.stride(0), q_exp.stride(1), q_exp.stride(2),
        k_exp.stride(0), k_exp.stride(1), k_exp.stride(2),
        v_f.stride(0),   v_f.stride(1),   v_f.stride(2),
        g_all.stride(0),    g_all.stride(1),
        beta_all.stride(0), beta_all.stride(1),
        # S_pre [N, NC, HV, K, V]
        S_pre.stride(0), S_pre.stride(1), S_pre.stride(2),
        S_pre.stride(3), S_pre.stride(4),
        # output [total_T, HV, V]
        output.stride(0), output.stride(1), output.stride(2),
        HV=HV, K=K, V=V, BK=BK, BV=BV, CS=CS, NC=NC,
        num_warps=4, num_stages=1,
    )

    return output, new_state
