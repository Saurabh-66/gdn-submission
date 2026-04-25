"""
GDN Prefill Kernel – MLSys 2026 FlashInfer Contest
Definition: gdn_prefill_qk4_v8_d128_k_last

Strategy: pure PyTorch, vectorized over all HV heads simultaneously.

The reference has 3 nested Python loops:
  for seq_idx:        # Python
    for token i:      # Python
      for h_idx:      # Python  <-- we eliminate this one

We keep the outer two loops (unavoidable — sequential state dependency)
but eliminate the innermost head loop entirely by batching all 8 heads
with PyTorch's bmm/matmul. This gives ~8x+ speedup over the reference
with zero compilation risk.
"""

import math
import torch
import torch.nn.functional as F


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

    # ── precompute g, beta for all tokens, all heads ──────────────────────
    x        = a.float() + dt_bias.float()
    g_all    = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [total_T, HV]
    beta_all = torch.sigmoid(b.float())                               # [total_T, HV]

    # ── GQA expansion: q,k from HQ heads -> HV heads ─────────────────────
    q_exp = q.float().repeat_interleave(GQA, dim=1)   # [total_T, HV, K]
    k_exp = k.float().repeat_interleave(GQA, dim=1)   # [total_T, HV, K]
    v_f   = v.float()                                  # [total_T, HV, V]

    if state is None:
        state_f = torch.zeros(num_seqs, HV, V, K, dtype=torch.float32, device=dev)
    else:
        state_f = state.float()

    # ── process each sequence ─────────────────────────────────────────────
    for seq_idx in range(num_seqs):
        t0    = int(cu_seqlens[seq_idx].item())
        t1    = int(cu_seqlens[seq_idx + 1].item())
        T_seq = t1 - t0
        if T_seq <= 0:
            new_state[seq_idx] = state_f[seq_idx]
            continue

        # State: [HV, V, K] k-last on disk -> work as S_KV [HV, K, V]
        S_KV = state_f[seq_idx].transpose(-1, -2).clone()  # [HV, K, V]

        q_s    = q_exp[t0:t1]    # [T_seq, HV, K]
        k_s    = k_exp[t0:t1]    # [T_seq, HV, K]
        v_s    = v_f  [t0:t1]    # [T_seq, HV, V]
        g_s    = g_all  [t0:t1]  # [T_seq, HV]
        beta_s = beta_all[t0:t1] # [T_seq, HV]

        # Sequential token scan, vectorized over HV heads
        for i in range(T_seq):
            g_i    = g_s[i]      # [HV]
            beta_i = beta_s[i]   # [HV]
            k_i    = k_s[i]      # [HV, K]
            v_i    = v_s[i]      # [HV, V]
            q_i    = q_s[i]      # [HV, K]

            # old_state = g * S_KV  [HV, K, V]
            old_state = g_i[:, None, None] * S_KV

            # old_v = k @ old_state  [HV, V]
            # [HV, 1, K] @ [HV, K, V] -> [HV, 1, V] -> [HV, V]
            old_v = (k_i.unsqueeze(1) @ old_state).squeeze(1)

            # new_v = beta*v + (1-beta)*old_v  [HV, V]
            new_v = beta_i[:, None] * v_i + (1.0 - beta_i[:, None]) * old_v

            # S_KV = old_state + k^T ⊗ (new_v - old_v)  [HV, K, V]
            # [HV, K, 1] * [HV, 1, V]
            S_KV = old_state + k_i.unsqueeze(2) * (new_v - old_v).unsqueeze(1)

            # output = scale * q @ S_KV  [HV, V]
            # [HV, 1, K] @ [HV, K, V] -> [HV, 1, V] -> [HV, V]
            o_i = scale * (q_i.unsqueeze(1) @ S_KV).squeeze(1)

            output[t0 + i] = o_i.to(torch.bfloat16)

        # Store new state [HV, K, V] -> [HV, V, K] k-last
        new_state[seq_idx] = S_KV.transpose(-1, -2)

    return output, new_state
