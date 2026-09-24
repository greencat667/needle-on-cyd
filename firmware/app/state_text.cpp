#include "state_text.h"

#include <stdio.h>

// Same buckets as tools/state_text.py (HUNGER, ENERGY, CURIOSITY).
const char* hunger_word(int v) {
    return v <= 20 ? "not hungry" : v <= 50 ? "a little hungry" : v <= 75 ? "hungry" : "starving";
}
const char* energy_word(int v) {
    return v <= 20 ? "exhausted" : v <= 40 ? "tired" : v <= 70 ? "rested" : "full of energy";
}
const char* curiosity_word(int v) {
    return v <= 30 ? "bored" : v <= 60 ? "a bit curious" : "very curious";
}

int state_text(const CreatureState& s, char* out, int cap) {
    return snprintf(out, cap, "I'm %s, %s and %s.", hunger_word(s.hunger), energy_word(s.energy),
                    curiosity_word(s.curiosity));
}
