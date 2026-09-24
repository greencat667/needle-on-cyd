"""Creature state -> the user-turn text Needle reads. Mirrors
firmware/app/state_text.cpp byte for byte (tests/test_state_text.py checks).

The numbers are the app's state; turning them into a sentence is the app's
job, the same way the Doom project (058) renders its game state as text.
Needle then decides from that text alone. No rule here picks a tool.
"""

HUNGER = [(20, "not hungry"), (50, "a little hungry"), (75, "hungry"), (100, "starving")]
ENERGY = [(20, "exhausted"), (40, "tired"), (70, "rested"), (100, "full of energy")]
CURIOSITY = [(30, "bored"), (60, "a bit curious"), (100, "very curious")]


def word(table, v):
    for top, w in table:
        if v <= top:
            return w
    return table[-1][1]


def render(state, template="words"):
    h, e, c = state["hunger"], state["energy"], state["curiosity"]
    if template == "numbers":
        return f"hunger: {h}\nenergy: {e}\ncuriosity: {c}"
    if template == "scores":
        return f"Hunger {h}/100, energy {e}/100, curiosity {c}/100. What should I do?"
    if template == "words":
        return f"I'm {word(HUNGER, h)}, {word(ENERGY, e)} and {word(CURIOSITY, c)}."
    raise ValueError(template)


TEMPLATES = ("numbers", "scores", "words")
