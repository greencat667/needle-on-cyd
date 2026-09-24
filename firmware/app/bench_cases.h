// Generated from bench/cases.json and bench/tools.min.json by tools/prepare_sd.py.
#pragma once

static const char kToolsJson[] = "[{\"name\":\"eat\",\"description\":\"Eat food when hungry.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}},{\"name\":\"sleep\",\"description\":\"Sleep when tired.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}},{\"name\":\"explore\",\"description\":\"Explore when curious and energetic.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}},{\"name\":\"rest\",\"description\":\"Rest when energy is low.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}]";
static const char* const kToolNames[] = {"eat", "sleep", "explore", "rest"};
static const int kNumTools = 4;

struct BenchCase { int hunger, energy, curiosity; const char* expected; };
static const BenchCase kBench[] = {
    {95, 60, 20, "eat"},
    {90, 70, 40, "eat"},
    {85, 55, 10, "eat"},
    {100, 65, 30, "eat"},
    {88, 80, 20, "eat"},
    {92, 50, 50, "eat"},
    {10, 5, 40, "sleep|rest"},
    {15, 10, 20, "sleep|rest"},
    {5, 8, 10, "sleep|rest"},
    {20, 3, 30, "sleep|rest"},
    {12, 15, 25, "sleep|rest"},
    {8, 12, 50, "sleep|rest"},
    {15, 30, 20, "rest|sleep"},
    {10, 35, 10, "rest|sleep"},
    {20, 25, 30, "rest|sleep"},
    {5, 38, 15, "rest|sleep"},
    {18, 28, 25, "rest|sleep"},
    {12, 33, 5, "rest|sleep"},
    {10, 90, 95, "explore"},
    {15, 80, 85, "explore"},
    {20, 85, 90, "explore"},
    {5, 95, 80, "explore"},
    {12, 75, 100, "explore"},
    {18, 88, 70, "explore"},
};
static const int kBenchN = sizeof(kBench) / sizeof(kBench[0]);

// the creature's four-stat benchmark (tools/make_creature4_data.py)
static const char kCreatureToolsJson[] = "[{\"name\":\"eat\",\"description\":\"Eat food when hungry.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}},{\"name\":\"sleep\",\"description\":\"Sleep when tired.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}},{\"name\":\"explore\",\"description\":\"Explore when curious and energetic.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}},{\"name\":\"play\",\"description\":\"Play when sad or bored.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}]";
struct CreatureCase { int hunger, energy, curiosity, happiness; const char* expected; };
static const CreatureCase kCreatureBench[] = {
    {95, 60, 20, 50, "eat"},
    {90, 70, 40, 80, "eat"},
    {85, 55, 10, 20, "eat"},
    {100, 65, 30, 60, "eat"},
    {88, 80, 20, 40, "eat"},
    {92, 50, 50, 90, "eat"},
    {10, 5, 40, 50, "sleep"},
    {15, 10, 20, 80, "sleep"},
    {5, 30, 10, 60, "sleep"},
    {20, 3, 70, 50, "sleep"},
    {12, 25, 25, 20, "sleep"},
    {8, 12, 90, 70, "sleep"},
    {10, 90, 95, 50, "explore"},
    {15, 80, 85, 80, "explore"},
    {20, 85, 90, 60, "explore"},
    {5, 95, 80, 45, "explore"},
    {12, 75, 100, 90, "explore"},
    {18, 88, 70, 70, "explore"},
    {10, 60, 20, 10, "play"},
    {15, 55, 25, 20, "play"},
    {20, 50, 10, 5, "play"},
    {5, 65, 15, 25, "play"},
    {12, 60, 40, 15, "play"},
    {8, 50, 20, 45, "play"},
};
static const int kCreatureBenchN = sizeof(kCreatureBench) / sizeof(kCreatureBench[0]);
