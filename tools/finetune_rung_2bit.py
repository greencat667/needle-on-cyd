#!/usr/bin/env python3
"""LoRA fine-tune on top of the shipped 2-bit rung, exported at 2 bits.

`needle finetune` starts from the half-precision master and exports at 4 bits
for every tensor. This keeps the post-trained 2-bit model instead:

  - the frozen base is the 2-bit L2 archive itself (weights dequantised from
    the file by the package's own reader), so training sees exactly what the
    device runs;
  - the package's training loop (finetune_local) runs unchanged, with its CQ
    straight-through quantiser set per tensor to the widths the archive uses
    (CQ4 for the embedding and mHC maps, CQ2 for every other matrix), which
    reproduces the base bit for bit at zero adapter;
  - export copies every tensor LoRA does not touch byte for byte from the
    2-bit archive, and re-packs only the adapted attention projections, at
    CQ2, with the exporter's own rotation / Lloyd-Max / packing steps.

    python tools/finetune_rung_2bit.py bench/creature_train_bal.jsonl \
        --base models/needle3-L2.cact --out models/needle3-L2-creature-2bit.cact
"""
import argparse, os, sys, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "vendor", "needle"))
os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import needle.model.run as run_mod  # noqa: E402
from needle.model import finetune as ft, quantize as qz  # noqa: E402
from needle.model.export import _nearest_idx, _pack_lsb, CQ  # noqa: E402
from cact_params import load as load_cact  # noqa: E402
import slice_cact  # noqa: E402

WIDE = ("embedding", "mhc_phi")  # CQ4 in the archive; every other matrix is CQ2


def cq_quantize_ls(w, bits, group_size=128):
    """quantize.cq_quantize with a least-squares group norm.

    Same format and the same reconstruction (codebook[idx] * norm @ H) and the
    same indices; only the stored norm differs: <rot, snapped> / |snapped|^2
    instead of |rot|. For a group that is already a CQ point this returns it
    unchanged (plain re-quantisation shrinks it by |codebook[idx]|, ~6%), and
    for any group it is the closest reconstruction with those indices.
    """
    cb = jnp.asarray(qz._cq_codebook_np(bits, group_size))
    D, g = w.shape[-1], group_size
    pad = (-D) % g
    wp = jnp.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, pad)]) if pad else w
    groups = wp.reshape(*wp.shape[:-1], -1, g).astype(jnp.float32)
    H = jnp.asarray(qz._cq_hadamard_np(g))
    rot = groups @ H
    norm = jnp.sqrt(jnp.sum(rot ** 2, axis=-1, keepdims=True))
    snapped = qz._cq_nearest(rot / jnp.maximum(norm, 1e-12), cb)
    ls = jnp.sum(rot * snapped, -1, keepdims=True) / jnp.maximum(jnp.sum(snapped ** 2, -1, keepdims=True), 1e-12)
    ls = ls.astype(jnp.float16).astype(jnp.float32)
    deq = ((snapped * ls) @ H).reshape(wp.shape).astype(w.dtype)
    return deq[..., :D] if pad else deq


def cq_ste_ls(w, bits, group_size=128):
    return w + jax.lax.stop_gradient(cq_quantize_ls(w, bits, group_size) - w)


def leaf_bits(path):
    name = qz.leaf_name(path)
    return 4 if any(w in name for w in WIDE) else 2


def cq_ste_by_leaf(params, bits_unused=None, group_size=128):
    """quantize.cq_ste_params with the archive's per-tensor widths."""
    def q(path, leaf):
        if not qz._is_quant_leaf(path, leaf):
            return leaf
        b = leaf_bits(path)
        if qz._reduces_second_last(path):
            return jnp.swapaxes(cq_ste_ls(jnp.swapaxes(leaf, -1, -2), b, group_size), -1, -2)
        return cq_ste_ls(leaf, b, group_size)
    return jax.tree_util.tree_map_with_path(q, params)


def cq_pack(w, bits, group=128):
    """export._cq_pack for any width (rotate, snap the direction, pack), storing
    the least-squares norm as cq_quantize_ls does."""
    cb = qz._cq_codebook_np(bits, group)
    H = qz._cq_hadamard_np(group)
    out, D = w.shape
    pad = (-D) % group
    wp = np.pad(w, ((0, 0), (0, pad))) if pad else w
    g = wp.reshape(out, -1, group).astype(np.float32)
    rot = g @ H
    norm = np.sqrt((rot ** 2).sum(-1, keepdims=True))
    unit = rot / np.maximum(norm, 1e-12)
    idx3 = _nearest_idx(unit, cb)                      # (out, groups, 128)
    snapped = cb[idx3]
    ls = (rot * snapped).sum(-1) / np.maximum((snapped ** 2).sum(-1), 1e-12)  # least-squares norm
    idx = idx3.reshape(out, -1)
    return _pack_lsb(idx, bits).tobytes() + ls.astype(np.float16).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--base", required=True, help="2-bit rung archive (tools/slice_cact.py)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--keep-head", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    params, cfg, _tok, _meta = load_cact(a.base)
    cfg.dtype = "float32"
    run_mod.load_checkpoint = lambda path, return_run=False: (
        (params, cfg, {}) if return_run else (params, cfg))
    qz.cq_ste_params = cq_ste_by_leaf  # finetune_local imports it from here at call time

    # zero adapter reproduces the archive: check one attention matrix
    k = np.asarray(params["stack"]["layers"]["block"]["self_attn"]["q_proj"]["kernel"][0])
    k2 = np.asarray(cq_ste_by_leaf({"stack": {"layers": {"block": {"self_attn": {"q_proj": {
        "kernel": params["stack"]["layers"]["block"]["self_attn"]["q_proj"]["kernel"]}}}}}})
        ["stack"]["layers"]["block"]["self_attn"]["q_proj"]["kernel"][0])
    print(f"  check     CQ2 re-quantise of the base: max |diff| {np.abs(k - k2).max():.2e}")

    adapter = os.path.splitext(a.out)[0] + ".lora.safetensors"
    args = types.SimpleNamespace(
        checkpoint=a.base, jsonl_path=a.data, generate=0, epochs=a.epochs, batch_size=a.batch,
        lr=a.lr, lora_rank=a.rank, lora_alpha=a.alpha, max_len=512, val_split=0.1, seed=0,
        checkpoint_dir=os.path.dirname(a.out) or ".", out=adapter, score=True)
    ft.finetune_local(args)

    # ---- export: the base archive's bytes, adapted attention re-packed at CQ2
    ad = ft.read_adapter(adapter)
    hdr, cb, ts = slice_cact.read(a.base)
    L = hdr[10]
    names = ["q_proj", "k_proj", "v_proj", "gate_proj", "out_proj"]
    slots = {"q_proj": 1, "k_proj": 2, "v_proj": 3, "gate_proj": 9, "out_proj": 10}
    sa = params["stack"]["layers"]["block"]["self_attn"]
    replaced = 0
    for key, v in ad["lora"].items():
        name = next(n for n in names if f"/{n}/" in key + "/")
        A, B = np.asarray(v["A"]), np.asarray(v["B"])            # (L, in, r), (L, r, out)
        base = np.asarray(sa[name]["kernel"])                     # (L, in, out)
        merged = base + ad["scale"] * np.matmul(A, B)
        for l in range(L):
            idx = 1 + 27 * l + slots[name]
            t = ts[idx]
            assert t.dtype == CQ and t.bits == 2 and t.shape == (merged.shape[2], merged.shape[1])
            ts[idx] = slice_cact.T(CQ, t.shape, cq_pack(merged[l].T, 2), 128, 2)
            replaced += 1
    if not a.keep_head:  # tuned locally: the head is not recalibrated; Needle drops it too
        i = 1 + 27 * L + 9 + 2 + 4 * hdr[31] + 1   # right after final_norm
        n_head = len(ts) - i - 1
        if n_head > 0:
            del ts[i:i + n_head]
            print(f"  dropped   the confidence head ({n_head} tensors)")
    hdr = list(hdr)
    hdr[1] = len(ts)
    n = slice_cact.write(a.out, hdr, cb, ts)
    print(f"  wrote     {a.out}  {n / 1e6:.2f} MB  {len(ts)} tensors  ({replaced} attention "
          f"matrices re-packed at CQ2, the rest byte-identical to {os.path.basename(a.base)})")


if __name__ == "__main__":
    main()
