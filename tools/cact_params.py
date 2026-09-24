"""Load a .cact archive back into the reference JAX model's parameter tree.

The weights come out of the archive itself (dequantised by the package's own
`read_export`), so the JAX forward run on them is the reference for exactly
the numbers the file holds: the 2-bit shipped rungs included.
This is the inverse of needle.model.export._tensors.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vendor", "needle"))

from needle.model.architecture import TransformerConfig, engram_geometry  # noqa: E402
from needle.model.export import read_export, parse_tokenizer_blob, RefTokenizer  # noqa: E402

BLOCK_KEYS = ["norm_in", "q_proj", "k_proj", "v_proj", "q_taps", "k_taps", "v_taps",
              "q_norm", "k_norm", "gate_proj", "out_proj", "post_norm", "attn_gate",
              "pre_hada", "d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b",
              "w3a", "w3b", "cond_v", "cond_u"]


def load(path):
    meta, ts = read_export(path)
    L = meta["num_layers"]
    cfg = TransformerConfig(
        vocab_size=meta["vocab_size"], d_model=meta["d_model"], num_heads=meta["num_heads"],
        num_kv_heads=meta["num_kv_heads"], num_layers=L, qk_head_dim=meta["qk_head_dim"],
        v_head_dim=meta["v_head_dim"], max_seq_len=meta["max_seq_len"],
        rope_theta=meta["rope_theta"], engram_orders=meta["engram_orders"],
        engram_slots=meta["engram_slots"], engram_layers=meta["engram_layers"],
        global_layers=meta["global_layers"], sliding_window=meta["sliding_window"],
        mhc_lanes=meta["mhc_lanes"], qkv_conv_taps=meta["qkv_conv_taps"],
        out_vocab=meta["out_vocab"] if meta["out_vocab"] != meta["vocab_size"] else 0,
        kv_window=meta["kv_window"], kv_bits=meta["kv_bits"], dtype="float32",
        flash=False, remat=False)
    i = 0
    emb = ts[i]; i += 1
    blocks = {k: [] for k in BLOCK_KEYS}
    for _ in range(L):
        for k in BLOCK_KEYS:
            blocks[k].append(ts[i]); i += 1
    st = {k: np.stack(v) for k, v in blocks.items()}
    T = lambda a: np.swapaxes(a, -1, -2)  # stored [out, in]; flax kernels are [in, out]
    block = {
        "ZCRMSNorm_0": {"scale": st["norm_in"]},
        "self_attn": {
            "q_proj": {"kernel": T(st["q_proj"])}, "k_proj": {"kernel": T(st["k_proj"])},
            "v_proj": {"kernel": T(st["v_proj"])},
            "q_taps": st["q_taps"], "k_taps": st["k_taps"], "v_taps": st["v_taps"],
            "q_norm": {"scale": st["q_norm"]}, "k_norm": {"scale": st["k_norm"]},
            "gate_proj": {"kernel": T(st["gate_proj"])}, "out_proj": {"kernel": T(st["out_proj"])},
        },
        "post_attn_norm": {"scale": st["post_norm"]},
        "attn_gate": st["attn_gate"].reshape(L),
        "pre_hada_norm": {"scale": st["pre_hada"]},
        "hadamard_mlp": {k: st[k] for k in ("d1", "d2", "b2", "d3", "d4", "w1a", "w1b",
                                            "w2a", "w2b", "w3a", "w3b", "cond_v", "cond_u")},
    }
    stack = {"layers": {"block": block}}
    lanes = meta["mhc_lanes"]
    for name in ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post", "mhc_b_res"):
        stack[name] = ts[i]; i += 1
    for name in ("mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"):
        phi = ts[i]; i += 1                       # (L*k, nC) -> (L, nC, k)
        k = phi.shape[0] // L
        stack[name] = phi.reshape(L, k, -1).transpose(0, 2, 1)
    i += 2                                        # hada perms: recomputed by the model
    params = {"embedding": {"embedding": emb}, "stack": stack}
    orders, heads, sub_dim = engram_geometry(cfg)
    ntab = len(orders) * heads
    for s in range(len(meta["engram_layers"])):
        tables, kp, vp, taps = ts[i:i + 4]; i += 4
        params[f"engrams_{s}"] = {
            "embedding": tables.reshape(ntab, cfg.engram_slots, sub_dim),
            "key_proj": {"kernel": kp.T}, "value_proj": {"kernel": vp.T}, "taps": taps}
    stack["final_norm"] = {"scale": ts[i]}; i += 1
    heads_out = {}
    if len(ts) - i - 1:
        codes = [int(c) for c in ts[i]]; i += 1
        keys = {1: "embedding_head", 2: "confidence_head", 3: "router_head"}
        for c in codes:
            probes, gain, query, row_bias, proj, bias = ts[i:i + 6]; i += 6
            l1, k = gain.shape
            heads_out[keys[c]] = {"probes": probes.reshape(l1, k, -1), "gain": gain,
                                  "query": query, "row_bias": row_bias,
                                  "proj": {"kernel": proj.T, "bias": bias}}
            if c == 3:
                heads_out[keys[c]]["calibration"] = ts[i]; i += 1
    tok = RefTokenizer(parse_tokenizer_blob(ts[i]))
    params.update(heads_out)
    return params, cfg, tok, meta
