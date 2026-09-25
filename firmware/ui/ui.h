// The CYD screens. Everything is drawn straight to the panel in regions.
#pragma once
#include <stdint.h>

#include "app/creature.h"
#include "app/state_text.h"

struct UiInfo {                 // shown on every screen that has room
    int layers;
    float model_mb;
    uint32_t free_kb;
    const char* model_label;    // e.g. "2 layers, creature-tuned"
};

struct UiStats {                // the debug screen
    int layers;
    float model_mb;
    uint32_t free_kb, min_free_kb, largest_kb;
    uint32_t peak_kb;
    float tok_per_s;
    int inferences;
    float avg_s;
    float last_s;
    uint64_t sd_bytes;
    uint32_t sd_reads;
    uint64_t flash_bytes;
};

enum UiHit { HIT_NONE, HIT_HUNGER, HIT_ENERGY, HIT_CURIOSITY, HIT_THINK, HIT_AGAIN, HIT_DEBUG,
             HIT_BACK, HIT_MODE };

// Cycle the 8 orientation settings, each labelled with its number.
void ui_orientation_test(int ms_each);
void ui_colour_test(int ms_each);
void ui_boot(const char* line1, const char* line2);
void ui_error(const char* title, const char* detail);
void ui_main(const CreatureState& s, const UiInfo& info);
void ui_main_value(const CreatureState& s, int row);   // redraw one row after a tap
UiHit ui_main_hit(int x, int y);
void ui_thinking(const CreatureState& s, const UiInfo& info);
void ui_thinking_progress(int phase, int step, const char* text);
void ui_decision(const char* tool, const char* reasoning, float conf, float seconds,
                 float tok_s, uint64_t sd_bytes, const UiInfo& info);
UiHit ui_decision_hit(int x, int y);
void ui_debug(const UiStats& st, const char* mode_name);
UiHit ui_debug_hit(int x, int y);   // HIT_MODE on the mode button, else HIT_BACK

// ---- the autonomous creature (stretch goal) ----
void ui_creature(const CreatureWorld& w, const UiInfo& info);           // whole screen
void ui_creature_stats(const CreatureWorld& w);                         // the four bars
void ui_creature_face(const CreatureWorld& w, uint32_t ms, bool thinking); // the creature, at time ms
void ui_creature_status(const char* big, uint16_t colour, const char* right, const char* why);
void ui_creature_bubble(const char* text);  // what it last said (and told Needle), along the bottom
void ui_creature_tally(int decisions, float last_s);  // "#12 in 37 s", top right
UiHit ui_creature_hit(int x, int y);
