#!/usr/bin/env python3
"""Fine-tuning data and a benchmark for the autonomous creature (stretch goal).

Four stats (hunger, energy, curiosity, happiness) and four tools (eat, sleep,
explore, play), in the same shape as make_creature_data.py: a small written
labelling policy applied to the words a sentence uses; sentences from
paraphrase families with shuffled clause order; the device's own template
("I'm X, Y, Z and W.") and the benchmark states held out of training.

    python tools/make_creature4_data.py --n 2400 --out bench/creature4_train.jsonl
"""
import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from state_text import HUNGER, ENERGY, CURIOSITY, word

HAPPINESS = [(30, "sad"), (65, "content"), (100, "happy")]
TOOLS = json.load(open(os.path.join(HERE, "..", "bench", "creature_tools.json")))

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
    "sad": ["sad", "unhappy", "a bit down", "lonely"],
    "content": ["content", "calm", "fine in myself", "okay in myself"],
    "happy": ["happy", "cheerful", "in a great mood", "joyful"],
}

FAMILIES = [
    lambda a, b, c, d: f"I feel {a}, {b}, {c} and {d}.",
    lambda a, b, c, d: f"Right now I am {a}. I'm also {b}, {c} and {d}.",
    lambda a, b, c, d: f"Status: {a}; {b}; {c}; {d}.",
    lambda a, b, c, d: f"The creature is {a}, {b}, {c} and {d}.",
    lambda a, b, c, d: f"Feeling {a} and {b}, {c}, and {d}.",
    lambda a, b, c, d: f"I'm {a}. {b[0].upper() + b[1:]}. {c[0].upper() + c[1:]}. {d[0].upper() + d[1:]}.",
]


def render_device(s):
    """The device template (firmware/app/state_text.cpp, creature_text)."""
    return (f"I'm {word(HUNGER, s['hunger'])}, {word(ENERGY, s['energy'])}, "
            f"{word(CURIOSITY, s['curiosity'])} and {word(HAPPINESS, s['happiness'])}.")


def policy(h, e, c, p):
    if h in ("hungry", "starving"):
        return "eat", h
    if e in ("exhausted", "tired"):
        return "sleep", e
    if p == "sad":
        return "play", p
    if c == "very curious" or (c == "a bit curious" and e == "full of energy"):
        return "explore", c
    return "play", "nothing urgent"


def cases():
    """24 clear-cut benchmark states, 6 per tool."""
    out = []
    add = lambda h, e, c, p, t: out.append({"state": {"hunger": h, "energy": e, "curiosity": c,
                                                      "happiness": p}, "expected": [t]})
    for h, e, c, p in [(95, 60, 20, 50), (90, 70, 40, 80), (85, 55, 10, 20), (100, 65, 30, 60),
                       (88, 80, 20, 40), (92, 50, 50, 90)]:
        add(h, e, c, p, "eat")
    for h, e, c, p in [(10, 5, 40, 50), (15, 10, 20, 80), (5, 30, 10, 60), (20, 3, 70, 50),
                       (12, 25, 25, 20), (8, 12, 90, 70)]:
        add(h, e, c, p, "sleep")
    for h, e, c, p in [(10, 90, 95, 50), (15, 80, 85, 80), (20, 85, 90, 60), (5, 95, 80, 45),
                       (12, 75, 100, 90), (18, 88, 70, 70)]:
        add(h, e, c, p, "explore")
    for h, e, c, p in [(10, 60, 20, 10), (15, 55, 25, 20), (20, 50, 10, 5), (5, 65, 15, 25),
                       (12, 60, 40, 15), (8, 50, 20, 45)]:
        add(h, e, c, p, "play")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    bench = cases()
    for c in bench:  # the benchmark must agree with the policy
        s = c["state"]
        t, _ = policy(word(HUNGER, s["hunger"]), word(ENERGY, s["energy"]),
                      word(CURIOSITY, s["curiosity"]), word(HAPPINESS, s["happiness"]))
        assert t in c["expected"], (s, t)
    json.dump(bench, open(os.path.join(HERE, "..", "bench", "creature_cases.json"), "w"), indent=1)
    held = {tuple(c["state"].values()) for c in bench}
    rows, counts, quota = [], {}, a.n // 4
    while len(rows) < a.n:
        s = {k: rng.randint(0, 100) for k in ("hunger", "energy", "curiosity", "happiness")}
        if tuple(s.values()) in held:
            continue
        words = (word(HUNGER, s["hunger"]), word(ENERGY, s["energy"]),
                 word(CURIOSITY, s["curiosity"]), word(HAPPINESS, s["happiness"]))
        tool, key = policy(*words)
        if counts.get(tool, 0) >= quota:
            continue
        parts = [rng.choice(SYN[w]) for w in words]
        surf = dict(zip(words, parts))
        order = parts[:]
        rng.shuffle(order)
        text = rng.choice(FAMILIES)(*order)
        if text.startswith("I'm ") and text.count(",") == 2 and " and " in text:
            continue  # the device template stays out of training
        reasoning = f"'{surf.get(key, key)}' -> {tool}"
        rows.append({"query": text, "tools": TOOLS, "reasoning": reasoning,
                     "answers": [{"name": tool, "arguments": {}}]})
        counts[tool] = counts.get(tool, 0) + 1
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} examples -> {a.out}  {counts}; benchmark: bench/creature_cases.json")


if __name__ == "__main__":
    main()
