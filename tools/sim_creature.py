#!/usr/bin/env python3
"""Simulate the creature's world (firmware/app/creature.cpp) with the policy the
creature model was trained to follow, to tune the drift rates: how often is it
sad / content / happy, hungry, tired, and which actions does it pick?"""
import random, sys
sys.path.insert(0, "tools")
from make_creature4_data import policy, HAPPINESS
from state_text import HUNGER, ENERGY, CURIOSITY, word

def run(p, hours=6, seed=1, think=38, act_len=25):
    rng = random.Random(seed)
    s = dict(h=30, e=70, c=40, p=60)
    moods, acts, t = {}, {}, 0.0
    def tick(dt, act):
        s['h'] += p['dh'] * dt; s['e'] -= p['de'] * dt; s['c'] += p['dc'] * dt; s['p'] -= p['dp'] * dt
        if act == 'eat': s['h'] -= p['eat'] * dt; s['p'] += 0.5 * dt
        if act == 'sleep': s['e'] += p['sleep'] * dt
        if act == 'explore': s['c'] -= p['exp'] * dt; s['p'] += p['exp_p'] * dt; s['e'] -= 1.0 * dt; s['h'] += 0.6 * dt
        if act == 'play': s['p'] += p['play'] * dt; s['e'] -= 1.0 * dt; s['c'] -= 0.5 * dt
        if rng.random() < 0.012 * dt:
            s['p'] += rng.choice([-15, 0, 10, -5, 20]); s['e'] += rng.choice([-10, 0, 5, -25, 5])
        for k in s: s[k] = min(100, max(0, s[k]))
    while t < hours * 3600:
        w = (word(HUNGER, round(s['h'])), word(ENERGY, round(s['e'])), word(CURIOSITY, round(s['c'])), word(HAPPINESS, round(s['p'])))
        moods[w[3]] = moods.get(w[3], 0) + 1
        a, _ = policy(*w)
        acts[a] = acts.get(a, 0) + 1
        if not p['pause']:
            for _ in range(think): tick(1, None); t += 1   # world keeps going while it thinks
        else:
            t += think
        for _ in range(act_len): tick(1, a); t += 1
    n = sum(moods.values())
    return {k: f"{v/n:.0%}" for k, v in sorted(moods.items())}, {k: f"{v/n:.0%}" for k, v in sorted(acts.items())}

OLD = dict(dh=0.8, de=0.5, dc=0.6, dp=0.4, eat=5, sleep=5, exp=5, exp_p=1.5, play=5, pause=True)
print("current:", run(OLD))
for name, P in [
    ("A world runs while thinking", dict(dh=0.35, de=0.3, dc=0.35, dp=0.5, eat=4, sleep=3.5, exp=3.5, exp_p=0.8, play=2.4, pause=False)),
    ("B", dict(dh=0.3, de=0.25, dc=0.4, dp=0.45, eat=4, sleep=3.5, exp=3.5, exp_p=0.8, play=2.0, pause=False)),
    ("C", dict(dh=0.25, de=0.22, dc=0.45, dp=0.4, eat=4, sleep=3.5, exp=3.0, exp_p=0.8, play=2.0, pause=False)),
    ("D", dict(dh=0.22, de=0.2, dc=0.5, dp=0.42, eat=4, sleep=3.5, exp=3.0, exp_p=0.6, play=1.8, pause=False)),
]:
    print(name, run(P))
