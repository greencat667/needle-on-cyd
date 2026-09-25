// Renders the firmware's own screens (firmware/ui/ui.cpp) into a software
// framebuffer, so screenshots match what the CYD draws pixel for pixel.
//   make -C host ui_preview && host/ui_preview out_dir
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../firmware/ui/display.h"
#include "../firmware/ui/ui.h"

static uint16_t fb[LCD_H][LCD_W];

bool lcd_init() { return true; }
bool lcd_present() { return true; }
void lcd_set_madctl(uint8_t) {}
static uint16_t strip_buf[LCD_W * 8];
uint16_t* lcd_strip() { return strip_buf; }
void lcd_push_strip(int x, int y, int w, int h) {
    for (int j = 0; j < h; j++) for (int i = 0; i < w; i++) fb[y + j][x + i] = lcd_unwire(strip_buf[j * w + i]);
}
void lcd_blit(int x, int y, int w, int h, const uint16_t* px) {
    for (int j = 0; j < h; j++) for (int i = 0; i < w; i++) if (x + i < LCD_W && y + j < LCD_H) fb[y + j][x + i] = px[j * w + i];
}
void lcd_set_invert(bool) {}
void lcd_fill(int x, int y, int w, int h, uint16_t c) {
    for (int j = y; j < y + h; j++)
        for (int i = x; i < x + w; i++)
            if (i >= 0 && j >= 0 && i < LCD_W && j < LCD_H) fb[j][i] = c;
}
void lcd_round_rect(int x, int y, int w, int h, int r, uint16_t c) {
    if (r * 2 > h) r = h / 2;
    if (r * 2 > w) r = w / 2;
    lcd_fill(x, y + r, w, h - 2 * r, c);
    for (int i = 0; i < r; i++) {
        int dy = r - i, dx = 0;
        while ((r - dx) * (r - dx) + dy * dy > r * r) dx++;
        lcd_fill(x + dx, y + i, w - 2 * dx, 1, c);
        lcd_fill(x + dx, y + h - 1 - i, w - 2 * dx, 1, c);
    }
}
void lcd_ellipse(int cx, int cy, int rx, int ry, uint16_t c) {
    for (int dy = -ry; dy <= ry; dy++) {
        float t = 1.0f - (float)(dy * dy) / (float)(ry * ry);
        int half = 0;
        while ((float)(half + 1) * (half + 1) <= t * rx * rx) half++;
        lcd_fill(cx - half, cy + dy, 2 * half + 1, 1, c);
    }
}
void lcd_circle(int cx, int cy, int r, uint16_t c) { lcd_ellipse(cx, cy, r, r, c); }
static const Glyph* find(const Font& f, char ch) {
    for (int i = 0; i < f.n; i++) if (f.glyphs[i].code == (uint8_t)ch) return &f.glyphs[i];
    return nullptr;
}
int lcd_text_width(const char* s, const Font& f) {
    int w = 0;
    for (; *s; s++) { const Glyph* g = find(f, *s); w += g ? g->adv : f.line_h / 3; }
    return w;
}
static uint16_t blend(uint16_t fg, uint16_t bg, int a) {
    int r = (((fg >> 11) & 31) * a + ((bg >> 11) & 31) * (15 - a)) / 15;
    int g = (((fg >> 5) & 63) * a + ((bg >> 5) & 63) * (15 - a)) / 15;
    int b = ((fg & 31) * a + (bg & 31) * (15 - a)) / 15;
    return (uint16_t)((r << 11) | (g << 5) | b);
}
int lcd_text(int x, int y, const char* s, const Font& f, uint16_t fg, uint16_t bg) {
    int x0 = x;
    for (; *s; s++) {
        const Glyph* g = find(f, *s);
        if (!g) { x += f.line_h / 3; continue; }
        const uint8_t* bits = f.bits + g->off;
        for (int i = 0; i < g->w * g->h; i++) {
            uint8_t b = bits[i >> 1];
            int a = (i & 1) ? (b & 15) : (b >> 4);
            int px = x + g->dx + i % g->w, py = y + g->dy + i / g->w;
            if (px >= 0 && py >= 0 && px < LCD_W && py < LCD_H) fb[py][px] = blend(fg, bg, a);
        }
        x += g->adv;
    }
    return x - x0;
}
void lcd_text_center(int x, int y, int w, const char* s, const Font& f, uint16_t fg, uint16_t bg) {
    lcd_text(x + (w - lcd_text_width(s, f)) / 2, y, s, f, fg, bg);
}
bool touch_init() { return false; }
bool touch_read(int*, int*) { return false; }
int g_touch_raw_x, g_touch_raw_y, g_touch_raw_z;
void led_init() {}
void led_set(LedColor) {}

static void save(const char* dir, const char* name) {
    char p[256];
    snprintf(p, sizeof(p), "%s/%s.ppm", dir, name);
    FILE* f = fopen(p, "wb");
    fprintf(f, "P6\n%d %d\n255\n", LCD_W, LCD_H);
    for (int y = 0; y < LCD_H; y++)
        for (int x = 0; x < LCD_W; x++) {
            uint16_t c = fb[y][x];
            unsigned char rgb3[3] = {(unsigned char)(((c >> 11) & 31) * 255 / 31),
                                     (unsigned char)(((c >> 5) & 63) * 255 / 63),
                                     (unsigned char)((c & 31) * 255 / 31)};
            fwrite(rgb3, 1, 3, f);
        }
    fclose(f);
}

// Frames over time: host/ui_preview --clip <dir> <action> <seconds> [fps]
static int clip(const char* dir, const char* act, float secs, int fps) {
    UiInfo info = {2, 8.6f, 9, "2 layers, creature, 2-bit"};
    CreatureWorld w;
    int a = !strcmp(act, "think") ? ACT_NONE : creature_action_of(act);
    const bool thinking = a == ACT_NONE;
    const float S[4][4] = {{85, 60, 30, 50}, {25, 12, 40, 55}, {15, 85, 88, 70}, {12, 60, 20, 22}};
    const float* st = a >= 0 ? S[a] : S[2];
    w.hunger = st[0]; w.energy = st[1]; w.curiosity = st[2]; w.happiness = st[3];
    w.action = a; w.action_left = a >= 0 ? 25 : 0;
    w.events = false;
    ui_creature(w, info);
    const char* names[4] = {"EAT", "SLEEP", "EXPLORE", "PLAY"};
    const char* why[4] = {"'hungry' -> eat", "'exhausted' -> sleep", "'very curious' -> explore",
                          "'sad' -> play"};
    char said[128];
    creature_text(w.snapshot(), thinking ? EV_FRIEND : -1, said, sizeof(said));
    ui_creature_bubble(said);
    ui_creature_tally(12, 37);
    int n = (int)(secs * fps);
    for (int f = 0; f < n; f++) {
        const float t = f / (float)fps;
        if (a >= 0) {
            w.tick(1.0f / fps);
            if (w.action_left < 1) w.action_left = 25;  // keep the action going for the clip
            char sm[16];
            snprintf(sm, sizeof(sm), "%.0f s", w.action_left);
            if (f % fps == 0) {
                ui_creature_stats(w);
                ui_creature_status(names[a], rgb(61, 220, 151), sm, why[a]);
            }
        } else if (f % fps == 0) {
            ui_creature_status("THINKING", rgb(255, 210, 63), nullptr, t < secs / 2 ? "reading..." : "'friend' -> play");
        }
        ui_creature_face(w, (uint32_t)(t * 1000), thinking);
        char name[64];
        snprintf(name, sizeof(name), "f%04d", f);
        save(dir, name);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 4 && !strcmp(argv[1], "--clip")) return clip(argv[2], argv[3], atof(argv[4]), argc > 5 ? atoi(argv[5]) : 30);
    const char* dir = argc > 1 ? argv[1] : ".";
    UiInfo info = {2, 8.6f, 10, "2 layers, creature, 2-bit"};
    struct { const char* name; float h, e, c, p; int act; float left; bool thinking; const char* big;
             const char* right; const char* why; int event; } S[] = {
        {"creature_explore", 15, 85, 88, 72, ACT_EXPLORE, 14, false, "EXPLORE", "14 s", "'very curious' -> explore", -1},
        {"creature_eat", 88, 60, 30, 55, ACT_EAT, 20, false, "EAT", "20 s", "'starving' -> eat", -1},
        {"creature_sleep", 20, 12, 40, 50, ACT_SLEEP, 18, false, "SLEEP", "18 s", "'exhausted' -> sleep", -1},
        {"creature_play", 12, 60, 20, 18, ACT_PLAY, 9, false, "PLAY", "9 s", "'sad' -> play", -1},
        {"creature_thinking", 40, 45, 70, 50, ACT_NONE, 0, true, "THINKING", nullptr, "reading...", -1},
        {"creature_speaking", 30, 35, 70, 80, ACT_NONE, 0, true, "THINKING", nullptr, "'friend' -> play", EV_FRIEND},
        {"creature_speaking_long", 90, 15, 70, 20, ACT_NONE, 0, true, "THINKING", nullptr, "'something tasty' -> eat", EV_TASTY},
    };
    for (auto& s : S) {
        CreatureWorld w;
        w.hunger = s.h; w.energy = s.e; w.curiosity = s.c; w.happiness = s.p;
        w.action = s.act; w.action_left = s.left;
        ui_creature(w, info);
        ui_creature_face(w, 1400, s.thinking);
        ui_creature_status(s.big, s.thinking ? rgb(255, 210, 63) : rgb(61, 220, 151), s.right, s.why);
        char said[128];
        creature_text(w.snapshot(), s.event, said, sizeof(said));
        ui_creature_bubble(said);
        ui_creature_tally(3, 37.2f);
        save(dir, s.name);
    }
    return 0;
}
