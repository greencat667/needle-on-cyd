#include "creature.h"

#include <stdio.h>
#include <string.h>

static float clampf(float v) { return v < 0 ? 0 : v > 100 ? 100 : v; }

static float frand(uint32_t* s) {  // xorshift32 in [0, 1)
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return (x >> 8) / 16777216.0f;
}

void CreatureWorld::tick(float dt) {
    // background drift. Rates tuned with tools/sim_creature.py (the model's
    // policy over 12 simulated hours): ~38% happy, 41% content, 21% sad, and
    // the four actions used about equally. The world keeps going while
    // Needle thinks, so a slow decision is felt.
    hunger += 0.25f * dt;
    energy -= 0.22f * dt;
    curiosity += 0.45f * dt;
    happiness -= 0.43f * dt;
    if (action_left > 0) {
        switch (action) {
            case ACT_EAT: hunger -= 4.0f * dt; happiness += 0.5f * dt; break;
            case ACT_SLEEP: energy += 3.5f * dt; break;
            case ACT_EXPLORE: curiosity -= 3.0f * dt; happiness += 0.8f * dt; energy -= 1.0f * dt;
                              hunger += 0.6f * dt; break;
            case ACT_PLAY: happiness += 3.2f * dt; energy -= 1.0f * dt; curiosity -= 0.5f * dt; break;
            default: break;
        }
        action_left -= dt;
        if (action_left <= 0) { action_left = 0; actions_done++; }
    }
    // now and then the world does something
    if (events && frand(&rng) < 0.012f * dt) {
        static const struct { const char* what; float dh, de, dc, dp; } E[] = {
            {"a storm rolls in", 0, -10, -10, -15},
            {"smells something tasty", 20, 0, 5, 0},
            {"a butterfly drifts past", 0, 5, 25, 10},
            {"a noisy night", 0, -25, 0, -5},
            {"a friend waves hello", 0, 5, 5, 20},
        };
        const auto& e = E[(int)(frand(&rng) * 5) % 5];
        hunger += e.dh; energy += e.de; curiosity += e.dc; happiness += e.dp;
        snprintf(event, sizeof(event), "%s", e.what);
    }
    hunger = clampf(hunger); energy = clampf(energy);
    curiosity = clampf(curiosity); happiness = clampf(happiness);
}

void CreatureWorld::start(int act, float seconds) {
    action = act;
    action_left = seconds;
}

CreatureState CreatureWorld::snapshot() const {
    return {(int)(hunger + 0.5f), (int)(energy + 0.5f), (int)(curiosity + 0.5f), (int)(happiness + 0.5f)};
}

const char* happiness_word(int v) { return v <= 30 ? "sad" : v <= 65 ? "content" : "happy"; }

int creature_text(const CreatureState& s, char* out, int cap) {
    return snprintf(out, cap, "I'm %s, %s, %s and %s.", hunger_word(s.hunger), energy_word(s.energy),
                    curiosity_word(s.curiosity), happiness_word(s.happiness));
}

int creature_action_of(const char* t) {
    if (!strcmp(t, "eat")) return ACT_EAT;
    if (!strcmp(t, "sleep")) return ACT_SLEEP;
    if (!strcmp(t, "explore")) return ACT_EXPLORE;
    if (!strcmp(t, "play")) return ACT_PLAY;
    return ACT_NONE;
}
