#!/usr/bin/env python3
"""Fine-tuning data and a benchmark for the creature, version 5: it says only
what stands out ("I'm starving and very curious.") and it notices what just
happened in the world ("A friend is waving hello! I'm a bit tired.").

Same shape as make_creature4_data.py: a small written labelling policy
applied to the words a sentence uses, paraphrase families, and the device's
own wording (firmware/app/creature.cpp, creature_text) plus the benchmark
states held out of training.

    python tools/make_creature5_data.py --n 3000 --out bench/creature5_train.jsonl
"""
import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from state_text import HUNGER, ENERGY, CURIOSITY, word

HAPPINESS = [(30, "sad"), (65, "content"), (100, "happy")]
TOOLS = json.load(open(os.path.join(HERE, "..", "bench", "creature_tools.json")))
# words the creature leaves out: nothing stands out about them
QUIET = {"not hungry", "a little hungry", "rested", "content"}

# what the world can do (firmware/app/creature.cpp), what the creature says,
# the key its reasoning quotes, and the action it prompts
EVENTS = {
    "friend": ("A friend is waving hello!", "friend", "play",
               ["A friend waves at me!", "My friend is saying hello!", "Someone I know is waving!",
                "A friend has come to visit!", "My best friend just turned up!"]),
    "tasty": ("I can smell something tasty!", "something tasty", "eat",
              ["Something smells delicious!", "There's a lovely smell of food!", "I smell something yummy!",
               "Mmm, what's that tasty smell?", "Somebody is cooking something nice!"]),
    "butterfly": ("A butterfly just drifted past!", "butterfly", "explore",
                  ["There's a butterfly!", "A butterfly flutters by!", "I just saw a butterfly!",
                   "Look, a butterfly!", "A butterfly landed nearby!"]),
    "storm": ("A storm is rolling in!", "storm", "sleep",
              ["It's starting to thunder!", "A storm is coming!", "Dark clouds are gathering!",
               "There's thunder and lightning!", "The wind is howling!"]),
    "noisy": ("That was a noisy night.", "noisy night", "sleep",
              ["I barely slept last night.", "It was so noisy all night.", "Something kept me up all night.",
               "The neighbours were loud all night.", "I hardly got a wink of sleep."]),
}

SYN = {
    "hungry": ["hungry", "really hungry", "very hungry", "ravenous"],
    "starving": ["starving", "famished", "starving hungry", "desperately hungry"],
    "exhausted": ["exhausted", "completely worn out", "about to fall asleep", "drained"],
    "tired": ["tired", "a bit tired", "low on energy", "weary"],
    "full of energy": ["full of energy", "bursting with energy", "wide awake", "energetic"],
    "bored": ["bored", "not curious", "uninterested", "not in the mood to explore"],
    "a bit curious": ["a bit curious", "slightly curious", "mildly curious", "a little curious"],
    "very curious": ["very curious", "eager to explore", "really curious", "itching to explore"],
    "sad": ["sad", "unhappy", "a bit down", "lonely"],
    "happy": ["happy", "cheerful", "in a great mood", "joyful"],
}
FINE = ["I'm fine.", "All good right now.", "Nothing much to report.", "I'm okay.", "Everything's fine."]


def join(ws):
    if len(ws) == 1:
        return ws[0]
    return ", ".join(ws[:-1]) + " and " + ws[-1]


def salient(s):
    ws = (word(HUNGER, s["hunger"]), word(ENERGY, s["energy"]),
          word(CURIOSITY, s["curiosity"]), word(HAPPINESS, s["happiness"]))
    return ws, [w for w in ws if w not in QUIET]


def render_device(s, event=None):
    """The device's wording (firmware/app/creature.cpp, creature_text)."""
    _, said = salient(s)
    feel = f"I'm {join(said)}." if said else "I feel fine."
    return f"{EVENTS[event][0]} {feel}" if event else feel


def policy(h, e, c, p, event):
    """Urgent needs first, then what just happened, then everyday needs."""
    if h == "starving":
        return "eat", h
    if e == "exhausted":
        return "sleep", e
    if event:
        _, key, tool, _ = EVENTS[event]
        return tool, key
    if h == "hungry":
        return "eat", h
    if e == "tired":
        return "sleep", e
    if p == "sad":
        return "play", p
    if c == "very curious" or (c == "a bit curious" and e == "full of energy"):
        return "explore", c
    if c == "bored":
        return "play", c
    return "play", "nothing urgent"


FAMILIES = [  # (feelings, event) -> text; the device's "<event> I'm ..." form is not among them
    lambda f, ev: f"I feel {f}." if not ev else f"{ev} I feel {f}.",
    lambda f, ev: f"Right now I'm {f}." if not ev else f"{ev} Right now I'm {f}.",
    lambda f, ev: f"The creature is {f}." if not ev else f"{ev} The creature is {f}.",
    lambda f, ev: f"Feeling {f}." if not ev else f"Feeling {f}. {ev}",
    lambda f, ev: f"I am {f}." if not ev else f"I am {f}. {ev}",
    lambda f, ev: f"Status: {f}." if not ev else f"{ev} Status: {f}.",
    # "I'm ..." openings like the device's, but never its exact form
    lambda f, ev: f"I'm feeling {f}." if not ev else f"{ev} I'm feeling {f}.",
    lambda f, ev: f"I'm {f} right now." if not ev else f"{ev} I'm {f} right now.",
    lambda f, ev: f"Well, I'm {f}." if not ev else f"{ev} Well, I'm {f}.",
    lambda f, ev: f"I'm {f} at the moment." if not ev else f"I'm {f} at the moment. {ev}",
]


def cases():
    """30 clear-cut states: 20 without an event (5 per tool) and 10 with one."""
    out = []
    add = lambda h, e, c, p, t, ev=None: out.append({"state": {"hunger": h, "energy": e, "curiosity": c,
                                                               "happiness": p}, "event": ev, "expected": [t]})
    for h, e, c, p in [(95, 60, 20, 50), (90, 70, 40, 80), (85, 55, 10, 20), (100, 65, 30, 60), (70, 60, 50, 50)]:
        add(h, e, c, p, "eat")
    for h, e, c, p in [(10, 5, 40, 50), (15, 10, 20, 80), (5, 30, 10, 60), (20, 3, 70, 50), (12, 35, 50, 50)]:
        add(h, e, c, p, "sleep")
    for h, e, c, p in [(10, 90, 95, 50), (15, 80, 85, 80), (20, 85, 90, 60), (5, 95, 50, 45), (12, 60, 100, 90)]:
        add(h, e, c, p, "explore")
    for h, e, c, p in [(10, 60, 20, 10), (15, 55, 25, 20), (20, 50, 10, 50), (5, 65, 15, 80), (12, 60, 40, 50)]:
        add(h, e, c, p, "play")
    add(10, 60, 40, 50, "play", "friend")
    add(30, 35, 50, 50, "play", "friend")      # tired, but a friend is waving
    add(90, 60, 40, 50, "eat", "friend")       # starving beats a friend
    add(10, 60, 90, 50, "eat", "tasty")
    add(20, 60, 40, 50, "explore", "butterfly")
    add(10, 10, 40, 50, "sleep", "butterfly")  # exhausted beats a butterfly
    add(15, 60, 90, 80, "sleep", "storm")
    add(10, 60, 40, 20, "sleep", "storm")
    add(10, 50, 50, 50, "sleep", "noisy")
    add(40, 80, 90, 50, "sleep", "noisy")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--event-share", type=float, default=0.4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    bench = cases()
    for c in bench:  # the benchmark must agree with the policy
        ws, _ = salient(c["state"])
        t, _ = policy(*ws, c["event"])
        assert t in c["expected"], (c, t)
    json.dump(bench, open(os.path.join(HERE, "..", "bench", "creature5_cases.json"), "w"), indent=1)
    held = {(tuple(c["state"].values()), c["event"]) for c in bench}
    device = {render_device(c["state"], c["event"]) for c in bench}
    rows, counts, quota = [], {}, a.n // 4
    while len(rows) < a.n:
        s = {k: rng.randint(0, 100) for k in ("hunger", "energy", "curiosity", "happiness")}
        ev = rng.choice(sorted(EVENTS)) if rng.random() < a.event_share else None
        if (tuple(s.values()), ev) in held:
            continue
        ws, said = salient(s)
        tool, key = policy(*ws, ev)
        if counts.get(tool, 0) >= quota:
            continue
        parts = [rng.choice(SYN[w]) for w in said]
        surf = dict(zip(said, parts))
        rng.shuffle(parts)
        ev_text = rng.choice(EVENTS[ev][3]) if ev else None
        if parts:
            text = rng.choice(FAMILIES)(join(parts), ev_text)
        else:
            text = rng.choice(FINE) if not ev else f"{ev_text} {rng.choice(FINE)}"
        if text in device:
            continue
        reasoning = f"'{surf.get(key, key)}' -> {tool}"
        rows.append({"query": text, "tools": TOOLS, "reasoning": reasoning,
                     "answers": [{"name": tool, "arguments": {}}]})
        counts[tool] = counts.get(tool, 0) + 1
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    evs = sum(1 for r in rows if any(r["query"].find(p) >= 0 for e in EVENTS.values() for p in e[3]))
    print(f"{len(rows)} examples ({evs} with an event) -> {a.out}  {counts}; benchmark: bench/creature5_cases.json")


if __name__ == "__main__":
    main()
