#!/usr/bin/env python3
"""Fine-tuning data for the creature's four tools, in Needle's local format.

Each example is a sentence describing how the creature feels and the one call
that fits it. Labels follow a small written policy (below), applied to the
words the sentence uses, so everything the label depends on is in the text
Needle reads. Sentences are drawn from several paraphrase families with
shuffled clause order; the device's own template (tools/state_text.py,
"words") is held out of training entirely, as are the 24 benchmark states, so
the benchmark measures a template the model never saw.

    python tools/make_creature_data.py --n 1200 --out bench/creature_train.jsonl
"""
import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from state_text import HUNGER, ENERGY, CURIOSITY, word, render

TOOLS = json.load(open(os.path.join(HERE, "..", "bench", "tools.json")))

# Synonyms per bucket (bucket keys are the device template's words).
SYN = {
    "not hungry": ["not hungry", "full", "not hungry at all", "well fed"],
    "a little hungry": ["a little hungry", "slightly hungry", "a bit peckish", "somewhat hungry"],
    "hungry": ["hungry", "really hungry", "very hungry", "ravenous"],
    "starving": ["starving", "famished", "starving hungry", "desperately hungry"],
    "exhausted": ["exhausted", "completely worn out", "about to fall asleep", "drained"],
    "tired": ["tired", "a bit tired", "low on energy", "weary"],
    "rested": ["rested", "okay for energy", "fairly energetic", "alert"],
    "full of energy": ["full of energy", "bursting with energy", "wide awake", "energetic"],
    "bored": ["bored", "not curious", "uninterested", "not in the mood to explore"],
    "a bit curious": ["a bit curious", "slightly curious", "mildly curious", "a little curious"],
    "very curious": ["very curious", "eager to explore", "really curious", "itching to explore"],
}

FAMILIES = [
    lambda a, b, c: f"I feel {a}, {b} and {c}.",
    lambda a, b, c: f"Right now I am {a}. I'm also {b}, and {c}.",
    lambda a, b, c: f"Status: {a}; {b}; {c}.",
    lambda a, b, c: f"The creature is {a}, {b} and {c}.",
    lambda a, b, c: f"Feeling {a} and {b}, and {c}.",
    lambda a, b, c: f"I'm {a}. {b[0].upper() + b[1:]}. {c[0].upper() + c[1:]}.",
]


def policy(h, e, c):
    """The labelling policy, on the words (the only thing the text carries)."""
    if h in ("hungry", "starving"):
        return "eat", f"'{h}' -> eat"
    if e == "exhausted":
        return "sleep", f"'{e}' -> sleep"
    if e == "tired":
        return "rest", f"'{e}' -> rest"
    if c == "very curious" or (c == "a bit curious" and e == "full of energy"):
        return "explore", f"'{c}' and '{e}' -> explore"
    return "rest", f"nothing urgent -> rest"


def sample_state(rng):
    return {"hunger": rng.randint(0, 100), "energy": rng.randint(0, 100),
            "curiosity": rng.randint(0, 100)}


def example(rng, state):
    hw, ew, cw = word(HUNGER, state["hunger"]), word(ENERGY, state["energy"]), word(CURIOSITY, state["curiosity"])
    tool, _ = policy(hw, ew, cw)
    parts = [rng.choice(SYN[hw]), rng.choice(SYN[ew]), rng.choice(SYN[cw])]
    rng.shuffle(parts)
    text = rng.choice(FAMILIES)(*parts)
    # reasoning quotes the surface words, as the base model's reasoning does
    surf = dict(zip((hw, ew, cw), parts)) if len({hw, ew, cw}) == 3 else {}
    key = {"eat": hw, "sleep": ew, "rest": ew, "explore": cw}[tool]
    quoted = surf.get(key, key)
    reasoning = f"'{quoted}' -> {tool}"
    return {"query": text, "tools": TOOLS, "reasoning": reasoning,
            "answers": [{"name": tool, "arguments": {}}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--balanced", action="store_true", help="equal examples per tool")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    bench = json.load(open(os.path.join(HERE, "..", "bench", "cases.json")))
    held_states = {tuple(c["state"].values()) for c in bench}
    held_texts = {render(c["state"], "words") for c in bench}
    rows, counts = [], {}
    quota = a.n // 4 if a.balanced else a.n  # equal examples per tool when balanced
    while len(rows) < a.n:
        s = sample_state(rng)
        if tuple(s.values()) in held_states:
            continue
        ex = example(rng, s)
        if ex["query"] in held_texts or ex["query"].startswith("I'm ") and " and " in ex["query"] \
                and ex["query"].count(",") == 1:
            continue  # keep the device template ("I'm X, Y and Z.") out of training
        t = ex["answers"][0]["name"]
        if counts.get(t, 0) >= quota:
            continue
        rows.append(ex)
        counts[t] = counts.get(t, 0) + 1
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} examples -> {a.out}  {counts}")


if __name__ == "__main__":
    main()
