#include "ui.h"

#include <stdio.h>
#include <string.h>

#include "display.h"
#include "fonts.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#ifndef LCD_MADCTL
#define LCD_MADCTL 0x00
#endif
#define LCD_MADCTL_RUNTIME LCD_MADCTL
#ifndef LCD_INVERT
#define LCD_INVERT 0
#endif
#define LCD_INVERT_RUNTIME LCD_INVERT

// Palette: night navy with the board's own yellow as the accent.
static const uint16_t BG = rgb(10, 14, 30);
static const uint16_t PANEL = rgb(22, 29, 54);
static const uint16_t LINE = rgb(40, 50, 86);
static const uint16_t TEXT = rgb(232, 236, 248);
static const uint16_t MUTED = rgb(138, 147, 178);
static const uint16_t YELLOW = rgb(255, 210, 63);
static const uint16_t GREEN = rgb(61, 220, 151);
static const uint16_t RED = rgb(255, 93, 93);
static const uint16_t CYAN = rgb(92, 200, 255);

static const int ROW_Y[3] = {58, 102, 146};
static const char* const ROW_LABEL[3] = {"HUNGER", "ENERGY", "CURIOSITY"};
static const uint16_t ROW_COLOR[3] = {rgb(255, 140, 90), rgb(61, 220, 151), rgb(92, 200, 255)};

static void header(const char* right) {
    lcd_fill(0, 0, LCD_W, 44, BG);
    lcd_text(14, 8, "NEEDLE", f_title, YELLOW, BG);
    lcd_text(14 + lcd_text_width("NEEDLE", f_title) + 8, 18, "ON ESP32", f_small, MUTED, BG);
    if (right) lcd_text(LCD_W - 12 - lcd_text_width(right, f_small), 18, right, f_small, MUTED, BG);
    lcd_fill(14, 42, LCD_W - 28, 1, LINE);
}

static void footer(const UiInfo& info) {
    char b[80];
    snprintf(b, sizeof(b), "%s  %.1f MB  %u KB free  no Wi-Fi", info.model_label, info.model_mb,
             (unsigned)info.free_kb);
    lcd_fill(0, LCD_H - 18, LCD_W, 18, BG);
    lcd_text_center(0, LCD_H - 16, LCD_W, b, f_small, MUTED, BG);
}

void ui_orientation_test(int ms_each) {
    static const uint8_t M[8] = {0x08, 0x48, 0x88, 0xC8, 0x28, 0x68, 0xA8, 0xE8};
    for (int i = 0; i < 8; i++) {
        lcd_set_madctl(M[i]);
        lcd_fill(0, 0, LCD_W / 2, LCD_H / 2, rgb(200, 40, 40));
        lcd_fill(LCD_W / 2, 0, LCD_W / 2, LCD_H / 2, rgb(40, 180, 60));
        lcd_fill(0, LCD_H / 2, LCD_W / 2, LCD_H / 2, rgb(40, 60, 200));
        lcd_fill(LCD_W / 2, LCD_H / 2, LCD_W / 2, LCD_H / 2, rgb(220, 200, 40));
        char b[24];
        snprintf(b, sizeof(b), "%d  TOP-LEFT", i + 1);
        lcd_text(10, 10, b, f_title, 0xFFFF, rgb(200, 40, 40));
        vTaskDelay(pdMS_TO_TICKS(ms_each));
    }
    lcd_set_madctl(LCD_MADCTL_RUNTIME);
}

// Cycle inversion x colour order; each frame labels itself 1..4 and shows
// named swatches, so the right setting is the one whose names match.
void ui_colour_test(int ms_each) {
    struct { bool inv; uint8_t mad; } C[4] = {{false, 0x08}, {false, 0x00}, {true, 0x08}, {true, 0x00}};
    const uint16_t sw[4] = {rgb(255, 0, 0), rgb(0, 255, 0), rgb(0, 0, 255), rgb(255, 255, 0)};
    const char* nm[4] = {"RED", "GREEN", "BLUE", "YELLOW"};
    for (int i = 0; i < 4; i++) {
        lcd_set_invert(C[i].inv);
        lcd_set_madctl(C[i].mad);
        lcd_fill(0, 0, LCD_W, LCD_H, 0x0000);
        char b[24];
        snprintf(b, sizeof(b), "%d  (black background)", i + 1);
        lcd_text(12, 10, b, f_title, 0xFFFF, 0x0000);
        for (int k = 0; k < 4; k++) {
            lcd_fill(12 + k * 76, 70, 68, 90, sw[k]);
            lcd_text(14 + k * 76, 170, nm[k], f_small, 0xFFFF, 0x0000);
        }
        vTaskDelay(pdMS_TO_TICKS(ms_each));
    }
    lcd_set_invert(LCD_INVERT_RUNTIME);
    lcd_set_madctl(LCD_MADCTL_RUNTIME);
}

void ui_boot(const char* line1, const char* line2) {
    lcd_fill(0, 0, LCD_W, LCD_H, BG);
    header(nullptr);
    lcd_text_center(0, 92, LCD_W, line1, f_body, TEXT, BG);
    if (line2) lcd_text_center(0, 122, LCD_W, line2, f_small, MUTED, BG);
}

void ui_error(const char* title, const char* detail) {
    lcd_fill(0, 0, LCD_W, LCD_H, BG);
    header(nullptr);
    lcd_text_center(0, 80, LCD_W, title, f_title, RED, BG);
    lcd_text_center(0, 124, LCD_W, detail, f_small, TEXT, BG);
}

static int row_value(const CreatureState& s, int row) {
    return row == 0 ? s.hunger : row == 1 ? s.energy : s.curiosity;
}
static const char* row_word(const CreatureState& s, int row) {
    return row == 0 ? hunger_word(s.hunger) : row == 1 ? energy_word(s.energy) : curiosity_word(s.curiosity);
}

void ui_main_value(const CreatureState& s, int row) {
    const int y = ROW_Y[row], x = 14, w = LCD_W - 28, h = 38;
    lcd_round_rect(x, y, w, h, 8, PANEL);
    lcd_text(x + 12, y + 4, ROW_LABEL[row], f_small, MUTED, PANEL);
    lcd_text(x + 12, y + 18, row_word(s, row), f_small, TEXT, PANEL);
    char v[8];
    snprintf(v, sizeof(v), "%d", row_value(s, row));
    lcd_text(x + w - 14 - lcd_text_width(v, f_title), y + 4, v, f_title, ROW_COLOR[row], PANEL);
    // bar
    const int bx = x + 128, bw = w - 128 - 64, by = y + 17;
    lcd_round_rect(bx, by, bw, 6, 3, LINE);
    int fw = bw * row_value(s, row) / 100;
    if (fw > 5) lcd_round_rect(bx, by, fw, 6, 3, ROW_COLOR[row]);
}

void ui_main(const CreatureState& s, const UiInfo& info) {
    lcd_fill(0, 0, LCD_W, LCD_H, BG);
    header("tap a value");
    for (int r = 0; r < 3; r++) ui_main_value(s, r);
    lcd_round_rect(90, 190, 140, 30, 15, YELLOW);
    lcd_text_center(90, 194, 140, "THINK", f_body, BG, YELLOW);
    lcd_text(LCD_W - 44, LCD_H - 34, "info", f_small, LINE, BG);
    footer(info);
}

UiHit ui_main_hit(int x, int y) {
    for (int r = 0; r < 3; r++)
        if (y >= ROW_Y[r] && y < ROW_Y[r] + 38) return (UiHit)(HIT_HUNGER + r);
    if (y >= 184 && y < 226 && x >= 70 && x < 250) return HIT_THINK;
    if (y < 44 || (x > LCD_W - 60 && y > LCD_H - 44)) return HIT_DEBUG;
    return HIT_NONE;
}

// ---- thinking ---------------------------------------------------------------
static int s_think_lines = 0;

void ui_thinking(const CreatureState& s, const UiInfo& info) {
    lcd_fill(0, 0, LCD_W, LCD_H, BG);
    header("local inference");
    lcd_text(14, 54, "THINKING...", f_title, YELLOW, BG);
    char b[96];
    state_text(s, b, sizeof(b));
    lcd_text(14, 88, b, f_small, TEXT, BG);
    lcd_round_rect(14, 110, LCD_W - 28, 92, 8, PANEL);
    lcd_text(24, 116, "<think>", f_small, MUTED, PANEL);
    s_think_lines = 0;
    footer(info);
}

// Stream the reasoning into the panel, wrapping at the panel width.
void ui_thinking_progress(int phase, int step, const char* text) {
    char b[64];
    if (phase == 1) {
        snprintf(b, sizeof(b), "reading the turn (%d tokens)", step);
        lcd_fill(24, 132, LCD_W - 48, 16, PANEL);
        lcd_text(24, 132, b, f_small, MUTED, PANEL);
        return;
    }
    if (phase == 2) {
        // last ~3 lines of reasoning, 38 chars each
        const int cols = 38;
        int n = (int)strlen(text);
        int start = n > cols * 3 ? n - cols * 3 : 0;
        lcd_fill(24, 132, LCD_W - 48, 52, PANEL);
        for (int l = 0; l < 3; l++) {
            int off = start + l * cols;
            if (off >= n) break;
            char line[40];
            int k = n - off < cols ? n - off : cols;
            memcpy(line, text + off, k);
            line[k] = 0;
            for (int i = 0; i < k; i++) if (line[i] == '\n') line[i] = ' ';
            lcd_text(24, 132 + l * 16, line, f_small, TEXT, PANEL);
        }
        return;
    }
    if (phase == 3) {
        lcd_fill(24, 184, LCD_W - 48, 16, PANEL);
        int n = (int)strlen(text);
        const char* t = n > 38 ? text + n - 38 : text;
        lcd_text(24, 184, t, f_small, CYAN, PANEL);
    }
}

// ---- decision ---------------------------------------------------------------
void ui_decision(const char* tool, const char* reasoning, float conf, float seconds, float tok_s,
                 uint64_t sd_bytes, const UiInfo& info) {
    lcd_fill(0, 0, LCD_W, LCD_H, BG);
    header("decided on-device");
    lcd_text(14, 50, "DECISION", f_small, MUTED, BG);
    char up[24];
    int i = 0;
    for (; tool[i] && i < 22; i++) up[i] = (tool[i] >= 'a' && tool[i] <= 'z') ? tool[i] - 32 : tool[i];
    up[i] = 0;
    if (!i) strcpy(up, "NOTHING");
    lcd_text_center(0, 64, LCD_W, up, f_huge, i ? GREEN : RED, BG);
    char b[96];
    snprintf(b, sizeof(b), "%s()", i ? tool : "[]");
    lcd_text_center(0, 118, LCD_W, b, f_small, CYAN, BG);
    if (reasoning && *reasoning) {
        char r[44];
        strncpy(r, reasoning, 40);
        r[40] = 0;
        if (strlen(reasoning) > 40) strcpy(r + 37, "...");
        lcd_text_center(0, 136, LCD_W, r, f_small, MUTED, BG);
    }
    lcd_round_rect(14, 156, LCD_W - 28, 28, 8, PANEL);
    snprintf(b, sizeof(b), "p %.2f   %.1f s   %.1f tok/s   %.1f MB read", conf, seconds, tok_s,
             sd_bytes / 1e6);
    lcd_text_center(14, 162, LCD_W - 28, b, f_small, TEXT, PANEL);
    lcd_round_rect(100, 194, 120, 26, 13, YELLOW);
    lcd_text_center(100, 197, 120, "AGAIN", f_body, BG, YELLOW);
    footer(info);
}

UiHit ui_decision_hit(int x, int y) {
    if (y >= 186 && y < 228 && x >= 80 && x < 240) return HIT_AGAIN;
    if (y < 44) return HIT_DEBUG;
    return HIT_NONE;
}

// ---- debug ------------------------------------------------------------------
void ui_debug(const UiStats& st, const char* mode_name) {
    lcd_fill(0, 0, LCD_W, LCD_H, BG);
    header("debug");
    char b[80];
    const char* rows[10];
    char buf[10][48];
    snprintf(buf[0], 48, "Needle layers        %d", st.layers);
    snprintf(buf[1], 48, "model size           %.2f MB", st.model_mb);
    snprintf(buf[2], 48, "free RAM             %u KB", (unsigned)st.free_kb);
    snprintf(buf[3], 48, "min free / largest   %u / %u KB", (unsigned)st.min_free_kb, (unsigned)st.largest_kb);
    snprintf(buf[4], 48, "runtime peak         %u KB", (unsigned)st.peak_kb);
    snprintf(buf[5], 48, "tokens/sec           %.2f", st.tok_per_s);
    snprintf(buf[6], 48, "inferences           %d", st.inferences);
    snprintf(buf[7], 48, "avg / last           %.1f / %.1f s", st.avg_s, st.last_s);
    snprintf(buf[8], 48, "SD read              %.1f MB, %u reads", st.sd_bytes / 1e6, (unsigned)st.sd_reads);
    snprintf(buf[9], 48, "flash read           %.1f MB", st.flash_bytes / 1e6);
    for (int i = 0; i < 10; i++) rows[i] = buf[i];
    for (int i = 0; i < 10; i++) lcd_text(18, 50 + i * 16, rows[i], f_small, i % 2 ? TEXT : CYAN, BG);
    lcd_round_rect(170, 212, 138, 24, 12, YELLOW);
    snprintf(b, sizeof(b), "switch to %s", mode_name);
    lcd_text_center(170, 216, 138, b, f_small, BG, YELLOW);
    lcd_text(18, 218, "tap to go back", f_small, MUTED, BG);
}

UiHit ui_debug_hit(int x, int y) { return (y >= 206 && x >= 160) ? HIT_MODE : HIT_BACK; }

// ---- the creature --------------------------------------------------------------
static const int PX = 8, PY = 50, PW = 152, PH = 146;  // its panel

static uint16_t mood_colour(float happiness) {
    return happiness > 65 ? rgb(255, 210, 63) : happiness > 30 ? rgb(120, 214, 140) : rgb(110, 150, 230);
}

// The creature panel is a tiny scene: a list of shapes, composed eight rows at
// a time into a 2.4 KB strip and pushed to the panel. Nothing is cleared on
// screen first, so it animates without flicker and without a framebuffer.
namespace {
enum { K_ELL, K_RECT, K_GLYPH };
struct Prim {
    uint8_t kind;
    int16_t x, y, a, b;   // ellipse: centre, radii; rect/glyph: top-left, size
    uint16_t c;
    const Glyph* g;
    const Font* f;
};
Prim g_prims[32];
int g_np = 0;

void ell(int cx, int cy, int rx, int ry, uint16_t c) {
    if (g_np < 32 && rx > 0 && ry > 0) g_prims[g_np++] = {K_ELL, (int16_t)cx, (int16_t)cy, (int16_t)rx, (int16_t)ry, c, nullptr, nullptr};
}
void rect(int x, int y, int w, int h, uint16_t c) {
    if (g_np < 32 && w > 0 && h > 0) g_prims[g_np++] = {K_RECT, (int16_t)x, (int16_t)y, (int16_t)w, (int16_t)h, c, nullptr, nullptr};
}
void glyph(int x, int y, char ch, const Font& f, uint16_t c) {
    for (int i = 0; i < f.n && g_np < 32; i++)
        if (f.glyphs[i].code == (uint8_t)ch) {
            const Glyph* g = &f.glyphs[i];
            g_prims[g_np++] = {K_GLYPH, (int16_t)(x + g->dx), (int16_t)(y + g->dy), g->w, g->h, c, g, &f};
            return;
        }
}
uint16_t mix(uint16_t fg, uint16_t bg, int a) {
    int r = (((fg >> 11) & 31) * a + ((bg >> 11) & 31) * (15 - a)) / 15;
    int g = (((fg >> 5) & 63) * a + ((bg >> 5) & 63) * (15 - a)) / 15;
    int b = ((fg & 31) * a + (bg & 31) * (15 - a)) / 15;
    return (uint16_t)((r << 11) | (g << 5) | b);
}
// the panel's own rounded corners
bool in_panel(int x, int y) {
    const int r = 10;
    int dx = x < PX + r ? PX + r - x : x > PX + PW - 1 - r ? x - (PX + PW - 1 - r) : 0;
    int dy = y < PY + r ? PY + r - y : y > PY + PH - 1 - r ? y - (PY + PH - 1 - r) : 0;
    return dx * dx + dy * dy <= r * r;
}
void render() {
    for (int y0 = PY; y0 < PY + PH; y0 += 8) {
        const int rows = PY + PH - y0 < 8 ? PY + PH - y0 : 8;
        for (int j = 0; j < rows; j++) {
            const int y = y0 + j;
            uint16_t* row = lcd_strip() + j * PW;  // composed in native order, swapped below
            for (int i = 0; i < PW; i++) row[i] = in_panel(PX + i, y) ? PANEL : BG;
            for (int k = 0; k < g_np; k++) {
                const Prim& p = g_prims[k];
                if (p.kind == K_ELL) {
                    const int dy = y - p.y;
                    if (dy < -p.b || dy > p.b) continue;
                    const float t = 1.0f - (float)(dy * dy) / (float)(p.b * p.b);
                    int half = 0;
                    while ((float)(half + 1) * (half + 1) <= t * p.a * p.a) half++;
                    for (int x = p.x - half; x <= p.x + half; x++)
                        if (x >= PX && x < PX + PW) row[x - PX] = p.c;
                } else if (p.kind == K_RECT) {
                    if (y < p.y || y >= p.y + p.b) continue;
                    for (int x = p.x; x < p.x + p.a; x++)
                        if (x >= PX && x < PX + PW) row[x - PX] = p.c;
                } else {  // glyph, alpha-blended over what is there
                    if (y < p.y || y >= p.y + p.b) continue;
                    const uint8_t* bits = p.f->bits + p.g->off;
                    for (int gx = 0; gx < p.a; gx++) {
                        const int x = p.x + gx;
                        if (x < PX || x >= PX + PW) continue;
                        const int idx = (y - p.y) * p.a + gx;
                        const uint8_t b = bits[idx >> 1];
                        const int a = (idx & 1) ? (b & 15) : (b >> 4);
                        if (a) row[x - PX] = mix(p.c, row[x - PX], a);
                    }
                }
            }
        }
        uint16_t* px = lcd_strip();
        for (int i = 0; i < PW * rows; i++) px[i] = lcd_wire(px[i]);
        lcd_push_strip(PX, y0, PW, rows);
    }
}
float tri(float t) { t -= (int)t; return t < 0.5f ? t * 2 : 2 - t * 2; }  // 0..1..0
}  // namespace

void ui_creature_face(const CreatureWorld& w, uint32_t ms, bool thinking) {
    g_np = 0;
    const float t = ms / 1000.0f;
    const bool acting = w.action_left > 0 && !thinking;
    const int act = acting ? w.action : ACT_NONE;
    const uint16_t body = mood_colour(w.happiness);
    const uint16_t ink = rgb(60, 40, 40);
    // movement by activity
    int cx = 70, cy = 128;
    float breathe = 0.5f + 0.5f * __builtin_sinf(t * (act == ACT_SLEEP ? 1.6f : 3.0f));
    if (act == ACT_PLAY) cy -= (int)(10 * tri(t * 1.1f));                 // hops
    if (act == ACT_EXPLORE) cx += (int)(22 * (tri(t * 0.25f) - 0.5f) * 2);  // wanders
    const int ry = 38 + (int)(3 * breathe), rx = 44 - (int)(2 * breathe);
    const int ground = PY + PH - 16;
    const int lift = ground - (cy + ry);
    ell(cx, ground, 40 - lift / 3 > 18 ? 40 - lift / 3 : 18, 5, rgb(16, 22, 42));  // shadow
    ell(cx - 30, cy - ry + 6, 8, 11, body);                                     // ears
    ell(cx + 30, cy - ry + 6, 8, 11, body);
    ell(cx, cy, rx, ry, body);
    // eyes: look where it is going, glance while thinking, blink now and then
    int lx = 0, ly = 0;
    if (thinking) { lx = (int)(4 * __builtin_sinf(t * 2.3f)); ly = -3; }
    else if (act == ACT_EXPLORE) lx = tri(t * 0.25f) < 0.5f ? 4 : -4;       // travel direction
    else if (act == ACT_PLAY) { lx = 4; ly = -2 + (int)(4 * tri(t * 1.1f)); }  // watching the ball
    const bool blink = !thinking && act != ACT_SLEEP && ((int)(t * 10) % 37) < 2;
    for (int s = -1; s <= 1; s += 2) {
        const int ex = cx + s * 16, ey = cy - 8;
        if (act == ACT_SLEEP || blink) {
            rect(ex - 7, ey, 14, 3, rgb(40, 40, 60));
        } else {
            ell(ex, ey, 9, 9, 0xFFFF);
            ell(ex + lx, ey + ly, 4, 4, rgb(20, 20, 30));
        }
    }
    // mouth
    if (act == ACT_EAT) {  // chewing
        const int open = 2 + (int)(5 * tri(t * 2.5f));
        ell(cx, cy + 16, 8, open, ink);
    } else if (w.happiness > 65 || act == ACT_PLAY) {
        ell(cx, cy + 12, 11, 7, ink);
        rect(cx - 12, cy + 4, 25, 8, body);
    } else if (w.happiness > 30) {
        rect(cx - 8, cy + 14, 16, 3, ink);
    } else {
        ell(cx, cy + 20, 10, 6, ink);
        rect(cx - 11, cy + 20, 23, 8, body);
    }
    // props, to the creature's right (never over its face)
    const int px = PX + PW - 22;
    if (thinking) {
        for (int i = 0; i < 3; i++) {
            const int on = ((int)(t * 3) % 3) == i;
            ell(cx + 22 + i * 12, cy - ry - 12, on ? 4 : 2, on ? 4 : 2, TEXT);
        }
    } else if (act == ACT_EAT) {  // an apple that loses bites
        const float left = w.action_left;
        const int bites = left > 18 ? 0 : left > 12 ? 1 : left > 6 ? 2 : 3;
        const int ay = ground - 12;
        if (bites < 3) {
            ell(px, ay, 10, 10, rgb(230, 60, 60));
            rect(px - 1, ay - 15, 3, 6, rgb(90, 60, 30));
            ell(px + 5, ay - 14, 5, 3, rgb(80, 200, 90));
            for (int b = 0; b < bites; b++) ell(px - 10 + b * 7, ay - 8 + b * 6, 5, 5, PANEL);
        }
    } else if (act == ACT_SLEEP) {  // z's drift up and away
        for (int i = 0; i < 3; i++) {
            float ph = t * 0.35f + i / 3.0f;
            ph -= (int)ph;
            const int zx = cx + 30 + (int)(ph * 22), zy = cy - ry - (int)(ph * 44);
            glyph(zx, zy - 10, i == 1 ? 'Z' : 'z', i == 1 ? f_title : f_body, TEXT);
        }
    } else if (act == ACT_EXPLORE) {  // magnifier held out in front
        const int mx = cx + (lx > 0 ? 46 : -46), my = cy - 4;
        ell(mx, my, 10, 10, CYAN);
        ell(mx, my, 6, 6, PANEL);
        rect(mx + (lx > 0 ? 5 : -8), my + 7, 4, 12, CYAN);
    } else if (act == ACT_PLAY) {  // a ball that really bounces: a falling arc, squashed on the ground
        float ph = t * 1.1f;
        ph -= (int)ph;
        const float h = 4 * ph * (1 - ph);                 // 0 at the ground, 1 at the top
        const int by = ground - 9 - (int)(70 * h);
        const bool squash = h < 0.06f;
        ell(px - 2, ground, 9 - (int)(h * 5), 3, rgb(16, 22, 42));  // its shadow
        ell(px - 2, by, squash ? 11 : 9, squash ? 7 : 9, rgb(255, 120, 60));
        rect(px - 11, by - 1, 18, 2, 0xFFFF);
    }
    render();
}

static const char* const STAT_NAME[4] = {"HUNGER", "ENERGY", "CURIOSITY", "HAPPINESS"};
static const uint16_t STAT_COL[4] = {rgb(255, 140, 90), rgb(61, 220, 151), rgb(92, 200, 255),
                                     rgb(255, 210, 63)};

void ui_creature_stats(const CreatureWorld& w) {
    const float v[4] = {w.hunger, w.energy, w.curiosity, w.happiness};
    for (int i = 0; i < 4; i++) {
        const int y = 52 + i * 27;
        lcd_fill(166, y, 146, 25, BG);
        lcd_text(166, y, STAT_NAME[i], f_small, MUTED, BG);
        char b[8];
        snprintf(b, sizeof(b), "%d", (int)(v[i] + 0.5f));
        lcd_text(312 - lcd_text_width(b, f_small), y, b, f_small, TEXT, BG);
        lcd_round_rect(166, y + 17, 146, 6, 3, LINE);
        int fw = (int)(146 * v[i] / 100);
        if (fw > 5) lcd_round_rect(166, y + 17, fw, 6, 3, STAT_COL[i]);
    }
}

void ui_creature_status(const char* big, const char* small, uint16_t colour) {
    lcd_round_rect(166, 158, 146, 40, 8, PANEL);
    lcd_text(174, 160, big, f_body, colour, PANEL);
    if (small) lcd_text(174, 180, small, f_small, MUTED, PANEL);
}

void ui_creature_line(const char* text) {
    lcd_fill(0, 200, LCD_W, 18, BG);
    char b[44];
    strncpy(b, text, 40);
    b[40] = 0;
    for (char* p = b; *p; p++) if (*p == '\n') *p = ' ';
    lcd_text_center(0, 202, LCD_W, b, f_small, CYAN, BG);
}

void ui_creature(const CreatureWorld& w, const UiInfo& info) {
    lcd_fill(0, 0, LCD_W, LCD_H, BG);
    header("creature");
    ui_creature_face(w, 0, false);
    ui_creature_stats(w);
    ui_creature_status("waking up", nullptr, TEXT);
    ui_creature_footer(info, 0, 0);
}

// Facts rather than a slogan: where it runs, the model, and how it is going.
void ui_creature_footer(const UiInfo& info, int decisions, float last_s) {
    char b[64];
    if (decisions)
        snprintf(b, sizeof(b), "on-device \xB7 Wi-Fi off \xB7 #%d in %.0f s", decisions, last_s);
    else
        snprintf(b, sizeof(b), "on-device \xB7 Wi-Fi off \xB7 %d layers", info.layers);
    lcd_fill(0, LCD_H - 18, LCD_W, 18, BG);
    lcd_text_center(0, LCD_H - 16, LCD_W, b, f_small, MUTED, BG);
}

UiHit ui_creature_hit(int, int y) { return y < 44 ? HIT_DEBUG : HIT_NONE; }
