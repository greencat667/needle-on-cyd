// The autonomous creature: a tiny world that drifts on its own, and an
// action Needle chose that pushes the stats back while it lasts. The world
// only describes itself (creature_text); the choice of action is Needle's.
#pragma once
#include <stdint.h>

#include "state_text.h"

enum CreatureAction { ACT_NONE = -1, ACT_EAT = 0, ACT_SLEEP = 1, ACT_EXPLORE = 2, ACT_PLAY = 3 };

struct CreatureWorld {
    float hunger = 30, energy = 70, curiosity = 40, happiness = 60;
    int action = ACT_NONE;
    float action_left = 0;     // seconds the current action still runs
    uint32_t actions_done = 0;
    uint32_t rng = 0x2545F491u;  // seeded from the hardware RNG at boot
    bool events = true;          // storms, butterflies... (off for rendered clips)
    char event[48] = "";       // last thing that happened in the world

    void tick(float dt);                   // advance the world by dt seconds
    void start(int act, float seconds);    // begin Needle's chosen action
    CreatureState snapshot() const;
};

const char* happiness_word(int v);
// "I'm hungry, tired, bored and sad." (tools/make_creature4_data.py render_device)
int creature_text(const CreatureState& s, char* out, int cap);
int creature_action_of(const char* tool_name);  // "eat" -> ACT_EAT, ...
