#!/usr/bin/env python3
"""LoRA fine-tune of one Needle 3 ladder rung, then export it as a .cact.

`needle finetune` trains the full 20-layer model and `needle build --layers N`
slices afterwards, so the adapter is never trained on the rung that ships.
This wrapper runs the package's own training loop (needle.model.finetune.
finetune_local: same LoRA targets, same CQ-W4 straight-through numerics, same
A8 activations) on the rung itself, by handing it the sliced checkpoint, and
then exports with the package's own writer. Nothing else is changed.

    python tools/finetune_rung.py bench/creature_train.jsonl --layers 2 \
        --epochs 4 --out models/needle3-L2-creature.cact
"""
import argparse, os, sys, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vendor", "needle"))
os.environ.setdefault("NEEDLE_TELEMETRY", "0")

import needle.model.run as run_mod  # noqa: E402
from needle.model import finetune as ft  # noqa: E402

CKPT = os.path.join(HERE, "..", "models", "checkpoints", "needle3.safetensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--keep-head", action="store_true",
                    help="keep the base confidence head (not trained here: uncalibrated)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    real_load = run_mod.load_checkpoint

    def sliced_load(path, return_run=False):
        got = real_load(path, return_run=return_run)
        params, config = got[0], got[1]
        params, config = ft.rung(params, config, a.layers)
        return (params, config, got[2]) if return_run else (params, config)

    run_mod.load_checkpoint = sliced_load
    adapter = os.path.splitext(a.out)[0] + ".lora.safetensors"
    args = types.SimpleNamespace(
        checkpoint=CKPT, jsonl_path=a.data, generate=0, epochs=a.epochs, batch_size=a.batch,
        lr=a.lr, lora_rank=a.rank, lora_alpha=a.alpha, max_len=512, val_split=0.1, seed=0,
        checkpoint_dir=os.path.dirname(a.out) or ".", out=adapter, score=True)
    ft.finetune_local(args)

    # export: rung params + merged adapter, the package's writer, CQ W4
    import jax.numpy as jnp
    from needle.model.architecture import ConfidenceHead, effective_kv_window
    from needle.model.export import read_tokenizer_blob, write_export
    from needle.model.checkpoints import read_checkpoint  # noqa: F401  (format check)
    params, config = sliced_load(CKPT)
    ad = ft.read_adapter(adapter)
    lora = {tuple(k.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])}
            for k, v in ad["lora"].items()}
    params = ft.merge_lora(params, lora, ad["scale"])
    if not a.keep_head and ConfidenceHead.key in params:
        params = {k: v for k, v in params.items() if k != ConfidenceHead.key}
        print("  dropped   the confidence head (not trained locally; Needle reports None)")
    base = os.path.join(HERE, "..", "models", "needle3.cact")
    info = write_export(params, config, a.out, tokenizer=read_tokenizer_blob(base),
                        kv_window=effective_kv_window(config))
    print(f"  wrote     {info['path']}  {info['bytes'] / 1e6:.2f} MB  {info['tensors']} tensors")


if __name__ == "__main__":
    main()
