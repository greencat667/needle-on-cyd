#pragma once
#include <stdint.h>

struct Glyph {
    uint16_t code;
    uint8_t w, h;     // ink box
    int8_t dx, dy;    // ink box offset from the pen position / line top
    uint8_t adv;      // horizontal advance
    uint32_t off;     // into the font's 4-bit bitmap
};

struct Font {
    const Glyph* glyphs;
    uint16_t n;
    const uint8_t* bits;
    uint8_t line_h;   // ascent + descent
    uint8_t ascent;
};
