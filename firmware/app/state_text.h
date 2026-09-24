// Creature state -> the sentence Needle reads (tools/state_text.py, "words").
#pragma once
struct CreatureState {
    int hunger, energy, curiosity, happiness;
};
// Writes e.g. "I'm starving, rested and bored." Returns length.
int state_text(const CreatureState& s, char* out, int cap);
const char* hunger_word(int v);
const char* energy_word(int v);
const char* curiosity_word(int v);
