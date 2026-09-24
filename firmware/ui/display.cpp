#include "display.h"

#include <string.h>

#include "board.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_attr.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "rom/ets_sys.h"
#include "sdkconfig.h"

#ifndef LCD_MADCTL
#define LCD_MADCTL 0x00  // RGB order, no row/column swap (chosen on this board by test frames)
#endif
#ifndef LCD_INVERT
#define LCD_INVERT 0     // no inversion (the colour test's frame 2)
#endif

#if CONFIG_NEEDLE_HEADLESS
bool lcd_init() { return false; }
bool lcd_present() { return false; }
void lcd_set_madctl(uint8_t) {}
void lcd_set_invert(bool) {}
void lcd_fill(int, int, int, int, uint16_t) {}
void lcd_blit(int, int, int, int, const uint16_t*) {}
static uint16_t s_dummy[LCD_W * 8];
uint16_t* lcd_strip() { return s_dummy; }
void lcd_push_strip(int, int, int, int) {}
void lcd_round_rect(int, int, int, int, int, uint16_t) {}
void lcd_circle(int, int, int, uint16_t) {}
void lcd_ellipse(int, int, int, int, uint16_t) {}
int lcd_text(int, int, const char*, const Font&, uint16_t, uint16_t) { return 0; }
int lcd_text_width(const char*, const Font&) { return 0; }
void lcd_text_center(int, int, int, const char*, const Font&, uint16_t, uint16_t) {}
bool touch_init() { return false; }
bool touch_read(int*, int*) { return false; }
int g_touch_raw_x = 0, g_touch_raw_y = 0, g_touch_raw_z = 0;
#else

static spi_device_handle_t s_lcd;
static bool s_ok = false;
static DMA_ATTR uint16_t s_px[LCD_W * 8];  // 5 KB strip buffer, reused by every draw

static void lcd_send(bool data, const void* p, int n) {
    if (n <= 0) return;
    gpio_set_level((gpio_num_t)PIN_TFT_DC, data);
    spi_transaction_t t = {};
    t.length = n * 8;
    t.tx_buffer = p;
    spi_device_polling_transmit(s_lcd, &t);
}
static void cmd(uint8_t c) { lcd_send(false, &c, 1); }
static void cmd_data(uint8_t c, const uint8_t* d, int n) {
    cmd(c);
    lcd_send(true, d, n);
}

static void window(int x, int y, int w, int h) {
    uint8_t ca[4] = {(uint8_t)(x >> 8), (uint8_t)x, (uint8_t)((x + w - 1) >> 8), (uint8_t)(x + w - 1)};
    uint8_t ra[4] = {(uint8_t)(y >> 8), (uint8_t)y, (uint8_t)((y + h - 1) >> 8), (uint8_t)(y + h - 1)};
    cmd_data(0x2A, ca, 4);
    cmd_data(0x2B, ra, 4);
    cmd(0x2C);
}

static inline uint16_t sw(uint16_t c) { return (uint16_t)((c >> 8) | (c << 8)); }

bool lcd_init() {
    gpio_config_t io = {};
    io.pin_bit_mask = (1ULL << PIN_TFT_DC) | (1ULL << PIN_TFT_BL);
    io.mode = GPIO_MODE_OUTPUT;
    gpio_config(&io);
    gpio_set_level((gpio_num_t)PIN_TFT_BL, 0);
    spi_bus_config_t bus = {};
    bus.mosi_io_num = PIN_TFT_MOSI;
    bus.miso_io_num = PIN_TFT_MISO;
    bus.sclk_io_num = PIN_TFT_SCLK;
    bus.quadwp_io_num = -1;
    bus.quadhd_io_num = -1;
    bus.max_transfer_sz = sizeof(s_px);
    if (spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_CH_AUTO) != ESP_OK) return false;
    spi_device_interface_config_t dev = {};
    dev.clock_speed_hz = 40 * 1000 * 1000;
    dev.mode = 0;
    dev.spics_io_num = PIN_TFT_CS;
    dev.queue_size = 1;
    if (spi_bus_add_device(SPI2_HOST, &dev, &s_lcd) != ESP_OK) return false;

    cmd(0x01);  // software reset (RST is tied to EN on the CYD)
    vTaskDelay(pdMS_TO_TICKS(120));
    // The CYD's panel wants the "ILI9341_2" settings (TFT_eSPI's CYD driver):
    // with the generic ILI9341 sequence the lower half of the screen garbles.
    static const uint8_t seq[] = {
        // cmd, n, data...
        0x28, 0,
        0xCF, 3, 0x00, 0x83, 0x30,
        0xED, 4, 0x64, 0x03, 0x12, 0x81,
        0xE8, 3, 0x85, 0x01, 0x79,
        0xCB, 5, 0x39, 0x2C, 0x00, 0x34, 0x02,
        0xF7, 1, 0x20,
        0xEA, 2, 0x00, 0x00,
        0xC0, 1, 0x26,
        0xC1, 1, 0x11,
        0xC5, 2, 0x35, 0x3E,
        0xC7, 1, 0xBE,
        0x36, 1, LCD_MADCTL,
        0x3A, 1, 0x55,
        0xB1, 2, 0x00, 0x1B,
        0xF2, 1, 0x08,
        0x26, 1, 0x01,
        0xE0, 15, 0x1F, 0x1A, 0x18, 0x0A, 0x0F, 0x06, 0x45, 0x87, 0x32, 0x0A, 0x07, 0x02, 0x07, 0x05, 0x00,
        0xE1, 15, 0x00, 0x25, 0x27, 0x05, 0x10, 0x09, 0x3A, 0x78, 0x4D, 0x05, 0x18, 0x0D, 0x38, 0x3A, 0x1F,
        0xB7, 1, 0x07,
        0xB6, 4, 0x0A, 0x82, 0x27, 0x00,
        0x00};
    lcd_set_invert(LCD_INVERT);
    for (const uint8_t* p = seq; *p;) {
        uint8_t c = *p++, n = *p++;
        cmd_data(c, p, n);
        p += n;
    }
    cmd(0x11);
    vTaskDelay(pdMS_TO_TICKS(120));
    cmd(0x29);
    s_ok = true;
    lcd_fill(0, 0, LCD_W, LCD_H, 0);
    gpio_set_level((gpio_num_t)PIN_TFT_BL, 1);
    return true;
}

bool lcd_present() { return s_ok; }

void lcd_set_madctl(uint8_t v) { cmd_data(0x36, &v, 1); }
void lcd_set_invert(bool on) { cmd(on ? 0x21 : 0x20); }

void lcd_fill(int x, int y, int w, int h, uint16_t c) {
    if (!s_ok || w <= 0 || h <= 0) return;
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > LCD_W) w = LCD_W - x;
    if (y + h > LCD_H) h = LCD_H - y;
    if (w <= 0 || h <= 0) return;
    const uint16_t s = sw(c);
    int total = w * h;
    int chunk = total < (int)(sizeof(s_px) / 2) ? total : (int)(sizeof(s_px) / 2);
    for (int i = 0; i < chunk; i++) s_px[i] = s;
    window(x, y, w, h);
    while (total > 0) {
        int n = total < chunk ? total : chunk;
        lcd_send(true, s_px, n * 2);
        total -= n;
    }
}

uint16_t* lcd_strip() { return s_px; }

void lcd_push_strip(int x, int y, int w, int h) {
    if (!s_ok || w * h > (int)(sizeof(s_px) / 2)) return;
    window(x, y, w, h);
    lcd_send(true, s_px, w * h * 2);
}

void lcd_blit(int x, int y, int w, int h, const uint16_t* px) {
    if (!s_ok || w <= 0 || h <= 0 || w * h > (int)(sizeof(s_px) / 2)) return;
    for (int i = 0; i < w * h; i++) s_px[i] = sw(px[i]);  // byte order for the wire
    window(x, y, w, h);
    lcd_send(true, s_px, w * h * 2);
}

void lcd_round_rect(int x, int y, int w, int h, int r, uint16_t c) {
    if (r * 2 > h) r = h / 2;
    if (r * 2 > w) r = w / 2;
    lcd_fill(x, y + r, w, h - 2 * r, c);
    for (int i = 0; i < r; i++) {  // cap rows: inset by the circle
        int dy = r - i;
        int dx = 0;
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
    for (int i = 0; i < f.n; i++)
        if (f.glyphs[i].code == (uint8_t)ch) return &f.glyphs[i];
    return nullptr;
}

int lcd_text_width(const char* s, const Font& f) {
    int w = 0;
    for (; *s; s++) {
        const Glyph* g = find(f, *s);
        w += g ? g->adv : f.line_h / 3;
    }
    return w;
}

static inline uint16_t blend(uint16_t fg, uint16_t bg, int a) {  // a in 0..15
    int r = (((fg >> 11) & 31) * a + ((bg >> 11) & 31) * (15 - a)) / 15;
    int g = (((fg >> 5) & 63) * a + ((bg >> 5) & 63) * (15 - a)) / 15;
    int b = ((fg & 31) * a + (bg & 31) * (15 - a)) / 15;
    return (uint16_t)((r << 11) | (g << 5) | b);
}

int lcd_text(int x, int y, const char* s, const Font& f, uint16_t fg, uint16_t bg) {
    if (!s_ok) return lcd_text_width(s, f);
    uint16_t pal[16];
    for (int a = 0; a < 16; a++) pal[a] = sw(blend(fg, bg, a));
    int x0 = x;
    for (; *s; s++) {
        const Glyph* g = find(f, *s);
        if (!g) { x += f.line_h / 3; continue; }
        if (g->w && g->h && (int)g->w * g->h <= (int)(sizeof(s_px) / 2)) {
            const uint8_t* bits = f.bits + g->off;
            int n = g->w * g->h;
            for (int i = 0; i < n; i++) {
                uint8_t b = bits[i >> 1];
                s_px[i] = pal[(i & 1) ? (b & 15) : (b >> 4)];
            }
            int gx = x + g->dx, gy = y + g->dy;
            if (gx >= 0 && gy >= 0 && gx + g->w <= LCD_W && gy + g->h <= LCD_H) {
                window(gx, gy, g->w, g->h);
                lcd_send(true, s_px, n * 2);
            }
        }
        x += g->adv;
    }
    return x - x0;
}

void lcd_text_center(int x, int y, int w, const char* s, const Font& f, uint16_t fg, uint16_t bg) {
    lcd_text(x + (w - lcd_text_width(s, f)) / 2, y, s, f, fg, bg);
}

// ---- XPT2046 touch, bit-banged ---------------------------------------------
#ifndef TOUCH_X_MIN
// calibrated on this board from taps at the four corners and the centre
#define TOUCH_X_MIN 285
#define TOUCH_X_MAX 3562
#define TOUCH_Y_MIN 441
#define TOUCH_Y_MAX 3799
#endif

int g_touch_raw_x = 0, g_touch_raw_y = 0, g_touch_raw_z = 0;

// Command byte, then 12 result bits clocked out MSB first; the controller
// shifts each bit on the falling edge, so sample after it.
static int tp_xfer(uint8_t c) {
    for (int i = 7; i >= 0; i--) {
        gpio_set_level((gpio_num_t)PIN_TP_MOSI, (c >> i) & 1);
        gpio_set_level((gpio_num_t)PIN_TP_CLK, 1);
        ets_delay_us(1);
        gpio_set_level((gpio_num_t)PIN_TP_CLK, 0);
        ets_delay_us(1);
    }
    gpio_set_level((gpio_num_t)PIN_TP_MOSI, 0);
    int v = 0;
    for (int i = 11; i >= 0; i--) {
        gpio_set_level((gpio_num_t)PIN_TP_CLK, 1);
        ets_delay_us(1);
        gpio_set_level((gpio_num_t)PIN_TP_CLK, 0);
        ets_delay_us(1);
        v |= gpio_get_level((gpio_num_t)PIN_TP_MISO) << i;
    }
    return v;
}

bool touch_init() {
    gpio_config_t o = {};
    o.pin_bit_mask = (1ULL << PIN_TP_CLK) | (1ULL << PIN_TP_MOSI) | (1ULL << PIN_TP_CS);
    o.mode = GPIO_MODE_OUTPUT;
    gpio_config(&o);
    gpio_config_t i = {};
    i.pin_bit_mask = (1ULL << PIN_TP_MISO) | (1ULL << PIN_TP_IRQ);
    i.mode = GPIO_MODE_INPUT;
    gpio_config(&i);
    gpio_set_level((gpio_num_t)PIN_TP_CS, 1);
    gpio_set_level((gpio_num_t)PIN_TP_CLK, 0);
    return true;
}

bool touch_read(int* x, int* y) {
    if (gpio_get_level((gpio_num_t)PIN_TP_IRQ)) return false;
    gpio_set_level((gpio_num_t)PIN_TP_CS, 0);
    int z1 = tp_xfer(0xB0), z2 = tp_xfer(0xC0);
    int z = z1 + 4095 - z2;
    long sx = 0, sy = 0;
    tp_xfer(0x90);  // discard the first conversion after the mux switches
    for (int k = 0; k < 4; k++) {
        sx += tp_xfer(0x90);
        sy += tp_xfer(0xD0);
    }
    gpio_set_level((gpio_num_t)PIN_TP_CS, 1);
    int rx = (int)(sx / 4), ry = (int)(sy / 4);
    g_touch_raw_x = rx;
    g_touch_raw_y = ry;
    g_touch_raw_z = z;
    if (z < 300) return false;
    int px = (rx - TOUCH_X_MIN) * LCD_W / (TOUCH_X_MAX - TOUCH_X_MIN);
    int py = (ry - TOUCH_Y_MIN) * LCD_H / (TOUCH_Y_MAX - TOUCH_Y_MIN);
#ifdef TOUCH_FLIP_X
    px = LCD_W - 1 - px;
#endif
#ifdef TOUCH_FLIP_Y
    py = LCD_H - 1 - py;
#endif
    *x = px < 0 ? 0 : px >= LCD_W ? LCD_W - 1 : px;
    *y = py < 0 ? 0 : py >= LCD_H ? LCD_H - 1 : py;
    return true;
}
#endif  // !CONFIG_NEEDLE_HEADLESS

// ---- RGB LED ---------------------------------------------------------------
void led_init() {
    gpio_config_t o = {};
    o.pin_bit_mask = (1ULL << PIN_LED_R) | (1ULL << PIN_LED_G) | (1ULL << PIN_LED_B);
    o.mode = GPIO_MODE_OUTPUT;
    gpio_config(&o);
    led_set(LED_OFF);
}

void led_set(LedColor c) {
    int r = 1, g = 1, b = 1;  // active low
    switch (c) {
        case LED_BLUE: b = 0; break;
        case LED_YELLOW: r = 0; g = 0; break;
        case LED_GREEN: g = 0; break;
        case LED_RED: r = 0; break;
        default: break;
    }
    gpio_set_level((gpio_num_t)PIN_LED_R, r);
    gpio_set_level((gpio_num_t)PIN_LED_G, g);
    gpio_set_level((gpio_num_t)PIN_LED_B, b);
}
