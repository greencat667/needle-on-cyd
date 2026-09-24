// ILI9341 (320x240 landscape) without a framebuffer: every draw call sets a
// window on the panel and streams just those pixels, so the panel's own GRAM
// is the only full-screen buffer. Headless builds compile to no-ops.
#pragma once
#include <stdint.h>

#include "font.h"

#define LCD_W 320
#define LCD_H 240

static inline uint16_t rgb(uint8_t r, uint8_t g, uint8_t b) {
    return (uint16_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
}

bool lcd_init();
bool lcd_present();
void lcd_set_madctl(uint8_t v);   // memory access control (orientation)
void lcd_set_invert(bool on);
void lcd_fill(int x, int y, int w, int h, uint16_t c);
// w*h RGB565 pixels (native order), e.g. a composed strip
void lcd_blit(int x, int y, int w, int h, const uint16_t* px);
// The driver's own DMA strip (LCD_W*8 pixels): compose into it with
// lcd_wire(colour) values, then lcd_push_strip. Saves a second buffer.
uint16_t* lcd_strip();
static inline uint16_t lcd_wire(uint16_t c) { return (uint16_t)((c >> 8) | (c << 8)); }
static inline uint16_t lcd_unwire(uint16_t c) { return (uint16_t)((c >> 8) | (c << 8)); }
void lcd_push_strip(int x, int y, int w, int h);
void lcd_round_rect(int x, int y, int w, int h, int r, uint16_t c);
void lcd_circle(int cx, int cy, int r, uint16_t c);
void lcd_ellipse(int cx, int cy, int rx, int ry, uint16_t c);
// Draw text with its top-left at (x, y) over a solid background; returns width.
int lcd_text(int x, int y, const char* s, const Font& f, uint16_t fg, uint16_t bg);
int lcd_text_width(const char* s, const Font& f);
// Centred in [x, x+w)
void lcd_text_center(int x, int y, int w, const char* s, const Font& f, uint16_t fg, uint16_t bg);

// Touch (XPT2046). Returns true while pressed, with screen coordinates.
bool touch_init();
bool touch_read(int* x, int* y);
extern int g_touch_raw_x, g_touch_raw_y, g_touch_raw_z;   // last raw ADC reading (calibration)

// RGB status LED (active low on the CYD)
enum LedColor { LED_OFF, LED_BLUE, LED_YELLOW, LED_GREEN, LED_RED };
void led_init();
void led_set(LedColor c);
