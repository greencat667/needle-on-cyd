"""Token-at-a-time NumPy forward of Needle 3, mirroring the C runtime step for step.

The JAX model in vendor/needle runs a whole sequence at once. The device can
only afford one token at a time with a KV cache, so this is the same maths
written incrementally: engram n-gram hashes over the token history, causal
conv taps over the last few q/k/v and engram values, an int8 KV cache, and the
multi-lane (mHC) residual stream. tools/check_ref.py pins it to the JAX
forward; tests/golden/*.json are dumped from it; firmware/needle_runtime is
its C translation.

Quantisation follows the reference forward run with quant=True (what
"Porting Needle 3", docs/upstream/SOURCES.md calls the engine's numerics): int8 per-vector
activations before every matmul, int8 per-head q/k/v, weights exactly as the
archive stores them.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vendor", "needle"))
from needle.model.export import read_export, parse_tokenizer_blob, RefTokenizer  # noqa: E402
from needle.model.architecture import _hada_perms  # noqa: E402

SEED, PRIME = 0x9E3779B9, 0x01000193
EPS = 1e-6
BLOCK = ["norm_in", "q_proj", "k_proj", "v_proj", "q_taps", "k_taps", "v_taps",
         "q_norm", "k_norm", "gate_proj", "out_proj", "post_norm", "attn_gate",
         "pre_hada", "d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b",
         "w3a", "w3b", "cond_v", "cond_u"]


def aq(x):
    """int8 absmax fake quant over the whole last axis (quantize.fake_quant, bits 8)."""
    x = x.astype(np.float32)
    amax = np.max(np.abs(x), axis=-1, keepdims=True)
    s = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    return (np.clip(np.round(x / s), -128, 127) * s).astype(np.float32)


def rms_unit(x):
    x = x.astype(np.float32)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)


def zcrms(x, scale):
    return (1.0 + scale) * rms_unit(x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def logsumexp(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))


def sinkhorn(logits, iters=20):
    k = logits
    for _ in range(iters):
        k = k - logsumexp(k, -1)
        k = k - logsumexp(k, -2)
    return np.exp(k)


class Model:
    def __init__(self, path):
        meta, ts = read_export(path)
        self.meta = meta
        g = lambda k: meta[k]
        self.L, self.D = g("num_layers"), g("d_model")
        self.H, self.KVH = g("num_heads"), g("num_kv_heads")
        self.qk, self.vd = g("qk_head_dim"), g("v_head_dim")
        self.lanes = g("mhc_lanes")
        self.taps = g("qkv_conv_taps")
        self.slots = g("engram_slots")
        self.sub = g("engram_sub_dim")
        self.ntab = g("num_engram_tables")
        self.etaps = g("engram_conv_taps")
        self.edil = g("engram_conv_dilation")
        self.orders = meta["engram_orders"]
        self.sites = meta["engram_layers"]
        self.globals = meta["global_layers"]
        self.window = g("sliding_window")
        self.theta = meta["rope_theta"]
        self.hada_n = g("hada_n")
        i = 0
        self.emb = ts[i]; i += 1
        self.blocks = []
        for _ in range(self.L):
            self.blocks.append(dict(zip(BLOCK, ts[i:i + 27]))); i += 27
        names = ["a_pre", "a_post", "a_res", "b_pre", "b_post", "b_res",
                 "phi_pre", "phi_post", "phi_res"]
        self.mhc = dict(zip(names, ts[i:i + 9])); i += 9
        self.p1 = ts[i].astype(np.int64); self.p2 = ts[i + 1].astype(np.int64); i += 2
        ref = _hada_perms(self.hada_n)
        assert (self.p1 == np.asarray(ref[0])).all() and (self.p2 == np.asarray(ref[1])).all()
        self.engram = []
        for _ in self.sites:
            tb, kp, vp, tp = ts[i:i + 4]; i += 4
            self.engram.append(dict(tables=tb, key=kp, value=vp, taps=tp))
        self.final_norm = ts[i]; i += 1
        self.heads = {}
        if len(ts) - i - 1:
            codes = [int(c) for c in ts[i]]; i += 1
            for c in codes:
                self.heads[c] = dict(zip(["probes", "gain", "query", "row_bias", "proj", "bias"],
                                         ts[i:i + 6])); i += 6
                if c == 3:
                    i += 1
        self.tok = RefTokenizer(parse_tokenizer_blob(ts[i]))
        heads_per_order = self.ntab // len(self.orders)
        self.stride = meta.get("engram_seed_heads") or heads_per_order
        self.heads_per_order = heads_per_order
        half = self.qk // 2
        self.inv_freq = (1.0 / (self.theta ** (np.arange(0, self.qk, 2, dtype=np.float32)
                                               / self.qk))).astype(np.float32)
        self.reset()

    # ---- state -------------------------------------------------------------
    def reset(self):
        self.t = 0
        self.tokens = []
        L = self.L
        self.kcache = [[] for _ in range(L)]      # per position: (int8 [KVH,qk], scale [KVH])
        self.vcache = [[] for _ in range(L)]
        self.qkv_hist = [[] for _ in range(L)]    # raw (pre-conv) q,k,v of recent positions
        self.ev_hist = [[] for _ in self.sites]   # engram value (pre-conv) history
        self.cells = []                           # hidden cells for the probe heads

    # ---- pieces -------------------------------------------------------------
    def engram_rows(self, s):
        """Fetch this token's engram rows for site s -> e (ntab*sub,)."""
        toks = self.tokens
        t = self.t
        rows = []
        for oi, order in enumerate(self.orders):
            for h in range(self.heads_per_order):
                seed = (SEED * (oi * self.stride + h + 1)) & 0xFFFFFFFF
                acc = seed
                for j in range(order):
                    tj = toks[t - j] if t - j >= 0 else 0
                    acc = ((acc ^ tj) * PRIME) & 0xFFFFFFFF
                acc ^= acc >> 15
                idx = acc % self.slots
                table = oi * self.heads_per_order + h
                ok = 1.0 if t >= order - 1 else 0.0
                rows.append(self.engram[s]["tables"][table * self.slots + idx] * ok)
        return np.concatenate(rows).astype(np.float32)

    def engram_kv(self, s):
        e = aq(self.engram_rows(s))
        eg = self.engram[s]
        k = eg["key"] @ e
        v = eg["value"] @ e
        self.ev_hist[s].append(v)
        out = np.zeros_like(v)
        for j in range(self.etaps):
            p = self.t - j * self.edil
            if p >= 0:
                out += eg["taps"][j] * self.ev_hist[s][p]
        return k, out

    def rope(self, x):
        ang = self.t * self.inv_freq
        c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)

    @staticmethod
    def q8_heads(x):
        """a8 per-head: returns (int8 codes, scales)."""
        amax = np.max(np.abs(x), -1, keepdims=True)
        s = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.round(x / s), -128, 127).astype(np.int8)
        return q, s[..., 0]

    def attention(self, l, x):
        b = self.blocks[l]
        H, KVH, qk, vd = self.H, self.KVH, self.qk, self.vd
        x = aq(x)
        q_raw, k_raw, v_raw = b["q_proj"] @ x, b["k_proj"] @ x, b["v_proj"] @ x
        hist = self.qkv_hist[l]
        hist.append((q_raw, k_raw, v_raw))
        q = np.zeros_like(q_raw); k = np.zeros_like(k_raw); v = np.zeros_like(v_raw)
        for j in range(self.taps):
            if self.t - j >= 0:
                qj, kj, vj = hist[self.t - j]
                q += b["q_taps"][j] * qj
                k += b["k_taps"][j] * kj
                v += b["v_taps"][j] * vj
        q = zcrms(q.reshape(H, qk), b["q_norm"])
        k = zcrms(k.reshape(KVH, qk), b["k_norm"])
        v = v.reshape(KVH, vd)
        q, k = self.rope(q), self.rope(k)
        qq, qs = self.q8_heads(q)
        kq, ks = self.q8_heads(k)
        vq, vs = self.q8_heads(v)
        self.kcache[l].append((kq, ks))
        self.vcache[l].append((vq, vs))
        is_global = l in self.globals
        lo = 0 if is_global or not self.window else max(0, self.t - self.window + 1)
        K = np.stack([c[0].astype(np.float32) * c[1][:, None] for c in self.kcache[l][lo:]])
        V = np.stack([c[0].astype(np.float32) * c[1][:, None] for c in self.vcache[l][lo:]])
        qf = qq.astype(np.float32) * qs[:, None]
        rep = H // KVH
        out = np.zeros((H, vd), np.float32)
        for h in range(H):
            kv = h // rep
            sc = K[:, kv, :] @ qf[h] / math.sqrt(qk)
            out[h] = softmax(sc) @ V[:, kv, :]
        out = out.reshape(-1) * sigmoid(b["gate_proj"] @ x)
        return b["out_proj"] @ aq(out)

    def hada_mlp(self, l, x):
        b = self.blocks[l]
        n = self.hada_n
        ba = b["w1a"].shape[0]
        bb = b["w1b"].shape[0]
        cond = 1.0 + softmax(x @ b["cond_v"]) @ b["cond_u"]
        z = np.zeros(n, np.float32); z[:self.D] = x

        def kron(z, a, c):
            return (a.T @ z.reshape(ba, bb) @ c).reshape(-1)

        z = kron(b["d1"] * z, b["w1a"], b["w1b"])[self.p1]
        z = kron(silu(b["d2"] * cond * z + b["b2"]), b["w2a"], b["w2b"])[self.p2]
        z = kron(b["d3"] * z, b["w3a"], b["w3b"])
        return (b["d4"] * z)[:self.D]

    def block(self, l, u, eng):
        b = self.blocks[l]
        if eng is not None:
            ek, ev = eng
            alpha = sigmoid(np.dot(rms_unit(u), rms_unit(ek)) / math.sqrt(self.D))
            u = u + alpha * ev
        skip = u
        h = self.attention(l, zcrms(u, b["norm_in"]))
        u = skip + sigmoid(b["attn_gate"][0]) * zcrms(h, b["post_norm"])
        skip = u
        return skip + self.hada_mlp(l, zcrms(u, b["pre_hada"]))

    # ---- one token -------------------------------------------------------
    def step(self, token, trace=None):
        """Advance one position; returns the final normalised hidden (D,)."""
        self.tokens.append(int(token))
        n, D = self.lanes, self.D
        x0 = self.emb[token] * math.sqrt(D)
        engs = [self.engram_kv(s) for s in range(len(self.sites))]
        X = np.broadcast_to(x0, (n, D)).astype(np.float32).copy()
        cells = [x0.astype(np.float32)]
        m = self.mhc
        if trace is not None:
            trace["x0"] = x0.copy()
            trace["engram"] = [(k.copy(), v.copy()) for k, v in engs]
        for l in range(self.L):
            lane = np.eye(n, dtype=np.float32)[l % n]
            pre_off, post_off = 8 * lane - 4, -4 * (1 - lane)
            nx = aq(rms_unit(X.reshape(-1)))
            phi_pre = m["phi_pre"][l * n:(l + 1) * n]
            phi_post = m["phi_post"][l * n:(l + 1) * n]
            phi_res = m["phi_res"][l * n * n:(l + 1) * n * n]
            hpre = sigmoid(m["a_pre"][l] * (phi_pre @ nx) + m["b_pre"][l] + pre_off)
            u = hpre @ X
            eng = None
            if l in self.sites:
                eng = engs[self.sites.index(l)]
            y = self.block(l, u, eng) - u
            hpost = 2 * sigmoid(m["a_post"][l] * (phi_post @ nx) + m["b_post"][l] + post_off)
            hres = sinkhorn(m["a_res"][l] * (phi_res @ nx).reshape(n, n) + m["b_res"][l])
            X = hres @ X + hpost[:, None] * y[None, :]
            cells.append(X.mean(0))
            if trace is not None:
                trace[f"layer{l}"] = X.copy()
        self.cells.append(np.stack(cells))
        h = zcrms(X.mean(0), self.final_norm)
        self.t += 1
        return h

    def logits(self, h):
        return self.emb @ aq(h)

    def confidence_logit(self, keep=None):
        """ConfidenceHead over the cells seen so far (probe_pool, then proj)."""
        hd = self.heads[2]
        cells = np.stack(self.cells)                    # (T, L+1, D)
        if keep is not None:
            cells = cells[keep]
        T, l1, D = cells.shape
        k = hd["gain"].shape[1]
        probes = hd["probes"].reshape(l1, k, D)
        scores = np.einsum("tld,lkd->lkt", cells, probes) / math.sqrt(D)
        r = np.einsum("lkt,tld->lkd", softmax(scores), cells)
        r = rms_unit(r) * hd["gain"][:, :, None]
        u = np.einsum("lkd,qd->qlk", r, hd["query"]) / math.sqrt(D) + hd["row_bias"]
        q = hd["query"].shape[0]
        w = softmax(u.reshape(q, l1 * k))
        pooled = (w @ r.reshape(l1 * k, D)).reshape(-1)
        return float((hd["proj"] @ pooled + hd["bias"])[0])
