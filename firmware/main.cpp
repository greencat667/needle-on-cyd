// Needle 3 on the Cheap Yellow Display: boot, storage, model, UI and serial.
//
// Everything below runs on the ESP32 itself. Wi-Fi and Bluetooth are never
// started; the only inputs are the touchscreen and the serial console.
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "app/bench_cases.h"
#include "app/creature.h"
#include "app/state_text.h"
#include "board.h"
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_image_format.h"
#include "esp_ota_ops.h"
#include "esp_random.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_rom_crc.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "inference.h"
#include "nr_platform.h"
#include "sdkconfig.h"
#include "storage/storage.h"
#include "ui/display.h"
#include "ui/ui.h"

// Two profiles on the card, each a folder with its own model, tool list,
// tokenizer index and prefix snapshot: the four-action demo (/needle) and
// the autonomous creature (/creature). /sdcard/profile.txt picks one.
static const char* PROFILE_PATH = SD_MOUNT "/profile.txt";
static const uint16_t YELLOW_UI = rgb(255, 210, 63), GREEN_UI = rgb(61, 220, 151),
                      RED_UI = rgb(255, 93, 93);
static bool g_creature = false;
static char MODEL_PATH[64], TOKIDX_PATH[64], SNAP_PATH[64], TOOLS_PATH[64], LABEL_PATH[64];

static void set_profile(bool creature) {
    g_creature = creature;
    const char* d = creature ? SD_MOUNT "/creature" : SD_DIR;
    snprintf(MODEL_PATH, sizeof(MODEL_PATH), "%s/needle3.cact", d);
    snprintf(TOKIDX_PATH, sizeof(TOKIDX_PATH), "%s/tok.idx", d);
    snprintf(SNAP_PATH, sizeof(SNAP_PATH), "%s/prefix.snap", d);
    snprintf(TOOLS_PATH, sizeof(TOOLS_PATH), "%s/tools.json", d);
    snprintf(LABEL_PATH, sizeof(LABEL_PATH), "%s/label.txt", d);
}

static FileReader g_sd;           // the archive on the SD card
static PartitionReader g_part;    // the `needle` flash partition
static OverlayReader g_rd(&g_sd); // archive reads, hot ranges from flash
static FileReader g_tokidx_file;
static OffsetReader g_tokidx;
static Model g_model;
static Tokenizer g_tok;
static NeedleSession g_sess;
static FileSnapshot g_snap;
static char g_tools[2048];
static char g_names[8][24];
static const char* g_name_ptrs[8];
static int g_ntools = 0;
static char g_label[48] = "";
static CreatureState g_state = {80, 35, 60, 50};

struct Totals {
    int inferences = 0;
    double seconds = 0, last = 0;
    uint32_t tokens = 0;
    double decode_s = 0;
} g_tot;

// ---------------------------------------------------------------------------
static uint32_t free_kb() { return (uint32_t)(esp_get_free_heap_size() / 1024); }

static void report_memory(const char* when) {
    nr_log("[mem] %s: free %u B, min free %u B, largest block %u B, runtime live %u B peak %u B"
           " (+%u B in IRAM; largest alloc %u B: %s)\n",
           when, (unsigned)esp_get_free_heap_size(), (unsigned)esp_get_minimum_free_heap_size(),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT),
           (unsigned)nr_mem()->current, (unsigned)nr_mem()->peak, (unsigned)nr_mem()->iram,
           (unsigned)nr_mem()->largest,
           nr_mem()->largest_tag ? nr_mem()->largest_tag : "-");
}

static uint32_t firmware_size() {
    const esp_partition_t* p = esp_ota_get_running_partition();
    esp_image_metadata_t md = {};
    esp_partition_pos_t pos = {p->address, p->size};
    if (esp_image_verify(ESP_IMAGE_VERIFY_SILENT, &pos, &md) != ESP_OK) return 0;
    return md.image_len;
}

static bool read_text(const char* path, char* dst, int cap) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int n = (int)fread(dst, 1, cap - 1, f);
    fclose(f);
    while (n > 0 && (dst[n - 1] == '\n' || dst[n - 1] == '\r' || dst[n - 1] == ' ')) n--;
    dst[n] = 0;
    return n > 0;
}

static void parse_tool_names() {
    g_ntools = 0;
    const char* p = g_tools;
    while (g_ntools < 8 && (p = strstr(p, "{\"name\":\""))) {
        p += 9;
        int i = 0;
        while (*p && *p != '"' && i < 23) g_names[g_ntools][i++] = *p++;
        g_names[g_ntools][i] = 0;
        g_name_ptrs[g_ntools] = g_names[g_ntools];
        g_ntools++;
    }
}

// Tokenizer surface index: built once in RAM before the model allocates,
// kept on SD, and read from there (2 bytes per probe) ever after.
static bool setup_tokenizer_index(uint32_t key) {
    uint32_t hdr[3];
    if (g_tokidx_file.open(TOKIDX_PATH, 1024) && g_tokidx_file.read(0, hdr, 12) &&
        hdr[0] == 0x58444954u && hdr[1] == key && g_tokidx_file.size() == 12 + hdr[2] * 2) {
        g_tokidx.set(&g_tokidx_file, 12);
        g_tok.use_stored_table(&g_tokidx, hdr[2]);
        nr_log("tokenizer: surface index from SD (%u entries)\n", (unsigned)hdr[2]);
        return true;
    }
    g_tokidx_file.close();
    nr_log("tokenizer: building the surface index (first boot)...\n");
    uint64_t t0 = nr_micros();
    if (!g_tok.open_lookup()) return false;
    FILE* f = fopen(TOKIDX_PATH, "wb");
    if (!f) return false;
    uint32_t entries = g_tok.table_bytes() / 2;
    uint32_t h[3] = {0x58444954u, key, entries};
    fwrite(h, 4, 3, f);
    fwrite(g_tok.table(), 2, entries, f);
    fclose(f);
    g_tok.close_lookup();
    nr_log("tokenizer: index built in %.1f s, saved to %s\n", (nr_micros() - t0) / 1e6, TOKIDX_PATH);
    return setup_tokenizer_index(key) || false;
}

static UiInfo ui_info() {
    UiInfo i;
    i.layers = (int)g_model.L;
    i.model_mb = g_sd.size() / 1e6f;
    i.free_kb = free_kb();
    i.model_label = g_label;
    return i;
}

static void on_progress(void*, int phase, int step, const char* text) {
    if (phase == 0) return;
    if (phase == 2) led_set((step & 1) ? LED_YELLOW : LED_OFF);
    ui_thinking_progress(phase, step, text);
}

static void print_result(const char* text, const NeedleResult& r) {
    const double s = r.total_us / 1e6;
    const double dec = r.decode_us / 1e6;
    printf("{\"input\":\"%s\",\"function_calls\":%s,\"reasoning\":\"", text, r.call_json);
    for (const char* p = r.reasoning; *p; p++) {
        if (*p == '"' || *p == '\\') putchar('\\');
        if (*p == '\n') { fputs("\\n", stdout); continue; }
        putchar(*p);
    }
    printf("\",\"confidence\":%.4f,\"call_prob\":%.4f,\"think_tokens\":%d,\"think_truncated\":%s,"
           "\"prefix_tokens\":%u,\"turn_tokens\":%u,\"generated\":%u,\"total_s\":%.2f,"
           "\"prefill_s\":%.2f,\"decode_s\":%.2f,\"decode_tok_s\":%.2f,\"bytes_read\":%llu,"
           "\"reads\":%u,\"peak_heap_used\":%u,\"largest_alloc\":%u,\"free_heap\":%u,"
           "\"min_free_heap\":%u}\n",
           r.confidence, r.call_prob, r.reasoning_tokens, r.think_truncated ? "true" : "false",
           (unsigned)r.prefix_tokens, (unsigned)r.turn_tokens, (unsigned)r.generated, s,
           r.prefill_us / 1e6, dec, dec > 0 ? r.generated / dec : 0.0,
           (unsigned long long)r.bytes_read, (unsigned)r.reads, (unsigned)nr_mem()->peak,
           (unsigned)nr_mem()->largest, (unsigned)esp_get_free_heap_size(),
           (unsigned)esp_get_minimum_free_heap_size());
}

static bool run_text(const char* text, NeedleResult* r, bool with_ui) {
    NeedleOptions opt;
    opt.think_budget = CONFIG_NEEDLE_THINK_BUDGET;
    opt.max_calls = 1;
    if (with_ui) opt.progress = on_progress;
    led_set(LED_YELLOW);
    uint64_t b_flash = g_part.bytes_read, f_us = g_part.read_us, c_bytes = g_sd.card_bytes;
    uint64_t c_us = g_sd.read_us, k_us = g_cq.us, k_macs = g_cq.macs;
    uint32_t c_reads = g_sd.card_reads;
    const ModelStats m0 = g_model.stats;
    bool ok = g_sess.run(text, opt, r);
    led_set(ok ? LED_GREEN : LED_RED);
    if (ok) {
        g_tot.inferences++;
        g_tot.last = r->total_us / 1e6;
        g_tot.seconds += g_tot.last;
        g_tot.tokens += r->generated;
        g_tot.decode_s += r->decode_us / 1e6;
        print_result(text, *r);
        nr_log("[io] this turn: SD %.1f MB in %u card reads, %.1f s; flash %.1f MB, %.1f s; "
               "matmul kernels %.1f s (incl. their reads), %.1f GMAC\n",
               (g_sd.card_bytes - c_bytes) / 1e6, (unsigned)(g_sd.card_reads - c_reads),
               (g_sd.read_us - c_us) / 1e6, (g_part.bytes_read - b_flash) / 1e6,
               (g_part.read_us - f_us) / 1e6, (g_cq.us - k_us) / 1e6, (g_cq.macs - k_macs) / 1e9);
        const ModelStats& m1 = g_model.stats;
        nr_log("[prof] restore %.2f s, encode %.2f s, forward %.1f s (engram %.1f, attention %.1f, "
               "mlp %.1f), output head %.1f s, %u positions\n",
               r->restore_us / 1e6, r->encode_us / 1e6, (m1.forward_us - m0.forward_us) / 1e6,
               (m1.engram_us - m0.engram_us) / 1e6, (m1.attn_us - m0.attn_us) / 1e6,
               (m1.mlp_us - m0.mlp_us) / 1e6, (m1.logits_us - m0.logits_us) / 1e6,
               (unsigned)(m1.tokens - m0.tokens));
    } else {
        nr_log("inference failed\n");
    }
    return ok;
}

static const char* tool_of(const NeedleResult& r) {
    return r.n_calls > 0 && r.call_tool[0] >= 0 ? g_names[r.call_tool[0]] : "";
}

static void think_ui() {
    ui_thinking(g_state, ui_info());
    char text[128];
    state_text(g_state, text, sizeof(text));
    NeedleResult r;
    if (!run_text(text, &r, true)) {
        ui_error("INFERENCE FAILED", "see the serial log");
        return;
    }
    const double dec = r.decode_us / 1e6;
    ui_decision(tool_of(r), r.reasoning, r.confidence, r.total_us / 1e6f,
                dec > 0 ? (float)(r.generated / dec) : 0.f, r.bytes_read, ui_info());
}

static void run_creature_bench();

static void run_bench() {
    if (g_creature) { run_creature_bench(); return; }
    int ok = 0;
    double t = 0;
    for (int i = 0; i < kBenchN; i++) {
        const BenchCase& c = kBench[i];
        CreatureState s = {c.hunger, c.energy, c.curiosity, 50};
        char text[128];
        state_text(s, text, sizeof(text));
        NeedleResult r;
        if (!run_text(text, &r, false)) continue;
        const char* tool = tool_of(r);
        bool hit = *tool && strstr(c.expected, tool) != nullptr;
        ok += hit;
        t += r.total_us / 1e6;
        printf("[bench] %2d/%d  h%3d e%3d c%3d  -> %-8s expected %-12s %s  %.1f s\n", i + 1, kBenchN,
               c.hunger, c.energy, c.curiosity, *tool ? tool : "[]", c.expected,
               hit ? "OK" : "MISS", r.total_us / 1e6);
    }
    printf("[bench] accuracy %d/%d, mean %.1f s per decision\n", ok, kBenchN, t / kBenchN);
}

static void run_creature_bench() {
    int ok = 0;
    double t = 0;
    for (int i = 0; i < kCreatureBenchN; i++) {
        const CreatureCase& c = kCreatureBench[i];
        CreatureState s = {c.hunger, c.energy, c.curiosity, c.happiness};
        char text[128];
        creature_text(s, c.event, text, sizeof(text));
        NeedleResult r;
        if (!run_text(text, &r, false)) continue;
        const char* tool = tool_of(r);
        bool hit = *tool && strstr(c.expected, tool) != nullptr;
        ok += hit;
        t += r.total_us / 1e6;
        printf("[bench] %2d/%d  \"%s\"  -> %-8s expected %-8s %s  %.1f s\n", i + 1, kCreatureBenchN, text,
               *tool ? tool : "[]", c.expected, hit ? "OK" : "MISS", r.total_us / 1e6);
    }
    printf("[bench] accuracy %d/%d, mean %.1f s per decision\n", ok, kCreatureBenchN, t / kCreatureBenchN);
}

static void print_stats() {
    nr_log("[stats] inferences %d, avg %.1f s, last %.1f s, decode %.2f tok/s, SD %llu B / %u reads,"
           " flash %llu B\n",
           g_tot.inferences, g_tot.inferences ? g_tot.seconds / g_tot.inferences : 0.0, g_tot.last,
           g_tot.decode_s > 0 ? g_tot.tokens / g_tot.decode_s : 0.0,
           (unsigned long long)g_sd.bytes_read, (unsigned)g_sd.reads,
           (unsigned long long)g_part.bytes_read);
    report_memory("now");
}

// ---------------------------------------------------------------------------
static char g_line[160];
static int g_line_n = 0;

// Receive a file over the serial console: "recv <path> <size>", then the
// host (tools/push_file.py) sends 512-byte chunks and waits for
// "ACK <n> <crc32>" after each. Written to <path>.part, renamed at the end.
static void receive_file(const char* path, uint32_t size, uint32_t baud) {
    static uint8_t chunk[512];
    char tmp[96];
    snprintf(tmp, sizeof(tmp), "%s.part", path);
    {   // create the parent folder if needed (e.g. /sdcard/creature)
        char dir[96];
        snprintf(dir, sizeof(dir), "%s", path);
        char* slash = strrchr(dir, '/');
        if (slash && slash != dir) { *slash = 0; mkdir(dir, 0775); }
    }
    FILE* f = fopen(tmp, "wb");
    if (!f) { printf("RECV-ERR open %s\n", tmp); return; }
    printf("RECV-READY %u\n", (unsigned)size);
    fflush(stdout);
    if (baud) {  // both ends switch for the transfer, then back to the console rate
        uart_wait_tx_done(UART_NUM_0, pdMS_TO_TICKS(100));
        uart_set_baudrate(UART_NUM_0, baud);
    }
    uint32_t got = 0, crc = 0;
    uint64_t t0 = nr_micros();
    while (got < size) {
        uint32_t want = size - got < sizeof(chunk) ? size - got : sizeof(chunk);
        int n = 0;
        while (n < (int)want) {
            int r = uart_read_bytes(UART_NUM_0, chunk + n, want - n, pdMS_TO_TICKS(3000));
            if (r <= 0) {
                fclose(f);
                remove(tmp);
                if (baud) uart_set_baudrate(UART_NUM_0, CONFIG_ESP_CONSOLE_UART_BAUDRATE);
                printf("RECV-ERR timeout at %u\n", (unsigned)got);
                return;
            }
            n += r;
        }
        crc = esp_rom_crc32_le(crc, chunk, want);
        if (fwrite(chunk, 1, want, f) != want) { fclose(f); remove(tmp); printf("RECV-ERR write\n"); return; }
        got += want;
        printf("ACK %u %08x\n", (unsigned)got, (unsigned)crc);
        fflush(stdout);
    }
    fclose(f);
    if (baud) {
        vTaskDelay(pdMS_TO_TICKS(200));
        uart_wait_tx_done(UART_NUM_0, pdMS_TO_TICKS(100));
        uart_set_baudrate(UART_NUM_0, CONFIG_ESP_CONSOLE_UART_BAUDRATE);
    }
    remove(path);
    if (rename(tmp, path) != 0) { printf("RECV-ERR rename\n"); return; }
    printf("RECV-DONE %s %u bytes crc %08x in %.1f s\n", path, (unsigned)size, (unsigned)crc,
           (nr_micros() - t0) / 1e6);
}

static void handle_command(char* cmd) {
    if (!strncmp(cmd, "recv ", 5)) {
        char path[80];
        unsigned size = 0, baud = 0;
        if (sscanf(cmd + 5, "%79s %u %u", path, &size, &baud) >= 2) receive_file(path, size, baud);
        return;
    }
    if (!strncmp(cmd, "mv ", 3)) {
        char a[80], b[80];
        if (sscanf(cmd + 3, "%79s %79s", a, b) == 2) {
            remove(b);
            printf(rename(a, b) == 0 ? "MV-OK\n" : "MV-ERR\n");
        }
        return;
    }
    if (!strcmp(cmd, "reboot")) esp_restart();
    if (!strncmp(cmd, "say ", 4)) {  // any text, straight to Needle
        NeedleResult r;
        run_text(cmd + 4, &r, false);
    } else if (!strncmp(cmd, "set ", 4)) {
        sscanf(cmd + 4, "%d %d %d", &g_state.hunger, &g_state.energy, &g_state.curiosity);
        think_ui();
    } else if (!strcmp(cmd, "think")) {
        think_ui();
    } else if (!strcmp(cmd, "bench")) {
        run_bench();
    } else if (!strcmp(cmd, "stats")) {
        print_stats();
    } else if (!strcmp(cmd, "colours")) {
        ui_colour_test(4000);
    } else if (!strcmp(cmd, "sdraw")) {
        storage_raw_bench();
    } else if (!strcmp(cmd, "sdbench")) {  // raw SD throughput, the model file
        static uint8_t buf[8192];
        uint64_t t0 = nr_micros();
        const uint32_t base = g_sd.size() / 2;
        for (uint32_t off = 0; off < 1024 * 1024; off += sizeof(buf)) g_sd.read(base + off, buf, sizeof(buf));
        double seq = (nr_micros() - t0) / 1e6;
        t0 = nr_micros();
        uint32_t x = 12345;
        for (int i = 0; i < 200; i++) {
            x = x * 1103515245u + 12345u;
            g_sd.read((x >> 4) % (g_sd.size() - 512), buf, 204);
        }
        double rnd = (nr_micros() - t0) / 1e6;
        t0 = nr_micros();
        for (int i = 0; i < 200; i++) {  // alternating two regions, like packed rows / norms
            g_sd.read(base + i * 4096, buf, 4096);
            g_sd.read(base + 3000000 + i * 256, buf, 256);
        }
        double alt = (nr_micros() - t0) / 1e6;
        printf("[sdbench] sequential 1 MB in 8 KB reads: %.2f s (%.2f MB/s); 200 random 204 B reads: "
               "%.1f ms each; 200 alternating 4 KB + 256 B pairs: %.1f ms per pair\n",
               seq, 1.0 / seq, rnd * 5, alt * 5);
    } else {
        printf("commands: say <text> | set <hunger> <energy> <curiosity> | think | bench | stats\n");
    }
}

static void poll_serial() {
    uint8_t c;
    while (uart_read_bytes(UART_NUM_0, &c, 1, 0) == 1) {
        if (c == '\r' || c == '\n') {
            if (g_line_n) {
                g_line[g_line_n] = 0;
                g_line_n = 0;
                handle_command(g_line);
            }
        } else if (g_line_n < (int)sizeof(g_line) - 1) {
            g_line[g_line_n++] = (char)c;
        }
    }
}

static void fail(const char* title, const char* detail) {
    led_set(LED_RED);
    nr_log("FATAL: %s: %s\n", title, detail);
    ui_error(title, detail);
    for (;;) {
        poll_serial();
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

static void show_debug() {
    UiStats st = {};
    st.layers = (int)g_model.L;
    st.model_mb = g_sd.size() / 1e6f;
    st.free_kb = free_kb();
    st.min_free_kb = (uint32_t)(esp_get_minimum_free_heap_size() / 1024);
    st.largest_kb = (uint32_t)(heap_caps_get_largest_free_block(MALLOC_CAP_8BIT) / 1024);
    st.peak_kb = (uint32_t)(nr_mem()->peak / 1024);
    st.tok_per_s = g_tot.decode_s > 0 ? (float)(g_tot.tokens / g_tot.decode_s) : 0;
    st.inferences = g_tot.inferences;
    st.avg_s = g_tot.inferences ? (float)(g_tot.seconds / g_tot.inferences) : 0;
    st.last_s = (float)g_tot.last;
    st.sd_bytes = g_sd.bytes_read;
    st.sd_reads = g_sd.reads;
    st.flash_bytes = g_part.bytes_read;
    ui_debug(st, g_creature ? "the demo" : "creature");
}

// The other profile needs its own model state: write the choice and restart.
static void switch_profile() {
    FILE* f = fopen(PROFILE_PATH, "w");
    if (f) {
        fputs(g_creature ? "needle\n" : "creature\n", f);
        fclose(f);
    }
    ui_boot("Switching...", g_creature ? "four-action demo" : "autonomous creature");
    vTaskDelay(pdMS_TO_TICKS(300));
    esp_restart();
}

// ---- the autonomous creature --------------------------------------------------
// The world drifts every second; when the creature's current action ends it
// describes itself in a sentence and Needle picks the next action. While
// Needle thinks, the per-position hook keeps the creature animated.
static CreatureWorld g_world;
static int g_frame = 0;

static char g_said[128] = "";  // the creature's last words (the bubble)
static char g_why[48] = "";    // Needle's reasoning for the current action
static uint64_t g_think_tick = 0;  // the world's clock while Needle thinks

static void creature_progress(void*, int phase, int, const char* text) {
    static uint64_t last = 0;
    const uint64_t now = nr_micros();
    char b[48];
    if (phase == 0 && now - g_think_tick >= 1000000) {  // the world goes on while it thinks
        g_world.tick((now - g_think_tick) / 1e6f);
        g_think_tick = now;
        ui_creature_stats(g_world);
    }
    if (phase == 0 && now - last > 250000) {  // one frame per forward position at most
        last = now;
        ui_creature_face(g_world, (uint32_t)(now / 1000), true);
        ++g_frame;
        led_set((g_frame & 1) ? LED_YELLOW : LED_OFF);
        snprintf(b, sizeof(b), "reading%.*s", g_frame % 4, "...");  // taking in what it said
        ui_creature_status("THINKING", YELLOW_UI, nullptr, b);
    } else if (phase == 2 && text) {
        ui_creature_status("THINKING", YELLOW_UI, nullptr, text);  // its reasoning, as it comes
    }
}

static const char* const ACT_NAME[4] = {"EAT", "SLEEP", "EXPLORE", "PLAY"};

static void creature_show_action() {
    char sm[16];
    snprintf(sm, sizeof(sm), "%.0f s", g_world.action_left);
    ui_creature_status(g_world.action >= 0 ? ACT_NAME[g_world.action] : "IDLE", GREEN_UI, sm, g_why);
}

static void creature_decide() {
    CreatureState s = g_world.snapshot();
    creature_text(s, g_world.pending, g_said, sizeof(g_said));
    g_world.pending = -1;
    g_world.event[0] = 0;
    g_world.quiet = true;  // one thing at a time: nothing new happens until it has decided
    g_think_tick = nr_micros();  // time before this was ticked by creature_loop
    ui_creature_bubble(g_said);  // what it tells Needle, said out loud
    ui_creature_status("THINKING", YELLOW_UI, nullptr, "Needle, on-device");
    ui_creature_face(g_world, (uint32_t)(nr_micros() / 1000), true);
    NeedleOptions opt;
    opt.think_budget = CONFIG_NEEDLE_THINK_BUDGET;
    opt.max_calls = 1;
    opt.progress = creature_progress;
    NeedleResult r;
    led_set(LED_YELLOW);
    bool ok = g_sess.run(g_said, opt, &r);
    led_set(ok ? LED_GREEN : LED_RED);
    g_world.quiet = false;
    if (!ok) {
        ui_creature_status("ERROR", RED_UI, nullptr, "see serial");
        g_world.start(ACT_NONE, 10);
        return;
    }
    g_tot.inferences++;
    g_tot.last = r.total_us / 1e6;
    g_tot.seconds += g_tot.last;
    g_tot.tokens += r.generated;
    g_tot.decode_s += r.decode_us / 1e6;
    print_result(g_said, r);
    const char* tool = tool_of(r);
    const int act = creature_action_of(tool);
    g_world.start(act, 25);
    snprintf(g_why, sizeof(g_why), "%.47s", r.reasoning);
    if (act < 0) {
        ui_creature_status("NOTHING", RED_UI, nullptr, g_why);
    } else {
        creature_show_action();
    }
    ui_creature_tally(g_tot.inferences, (float)g_tot.last);
}

static void creature_loop() {
    g_world.rng = esp_random() | 1u;  // a different life every boot
    ui_creature(g_world, ui_info());
    uint64_t last = nr_micros(), last_frame = 0;
    bool was_down = false;
    bool debug = false;
    creature_decide();
    for (;;) {
        poll_serial();
        const uint64_t now = nr_micros();
        if (!debug && now - last_frame >= 110000) {  // ~8 frames a second of animation
            last_frame = now;
            ui_creature_face(g_world, (uint32_t)(now / 1000), false);
        }
        if (!debug && now - last >= 1000000) {
            const float dt = (now - last) / 1e6f;
            last = now;
            g_world.tick(dt);
            ui_creature_stats(g_world);
            if (g_world.action_left <= 0) {  // done, or something happened: it speaks, Needle decides
                creature_decide();
                last = nr_micros();  // the world ticked inside creature_progress
            } else {
                creature_show_action();
            }
        }
        int x, y;
        bool down = touch_read(&x, &y);
        if (down && !was_down) {
            if (debug) {
                if (ui_debug_hit(x, y) == HIT_MODE) switch_profile();
                debug = false;
                ui_creature(g_world, ui_info());
                if (*g_said) ui_creature_bubble(g_said);
                if (g_tot.inferences) ui_creature_tally(g_tot.inferences, (float)g_tot.last);
                last = nr_micros();
            } else if (ui_creature_hit(x, y) == HIT_DEBUG) {
                show_debug();
                debug = true;
            }
        }
        was_down = down;
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

extern "C" void app_main(void) {
    led_init();
    led_set(LED_BLUE);
    uart_driver_install(UART_NUM_0, 512, 0, 0, nullptr, 0);
    lcd_init();
#ifdef LCD_COLOUR_TEST
    ui_colour_test(LCD_COLOUR_TEST);
#endif
#ifdef LCD_TEST_PATTERN
    ui_orientation_test(LCD_TEST_PATTERN);
#endif
    touch_init();
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    nr_log("\n=== Needle 3 on ESP32 (CYD) ===\n");
    nr_log("[boot] chip rev %d, %d cores, firmware %u B, Wi-Fi/BT never started\n", chip.revision,
           chip.cores, (unsigned)firmware_size());
    report_memory("boot");
    ui_boot("Mounting the SD card...", nullptr);

    if (!storage_mount_sd()) fail("NO SD CARD", "insert a card prepared by tools/prepare_sd.py");
    {
        char prof[16] = "";
        read_text(PROFILE_PATH, prof, sizeof(prof));
        set_profile(!strcmp(prof, "creature"));
        nr_log("[boot] profile: %s\n", g_creature ? "creature" : "four-action demo");
    }
    if (!g_sd.open(MODEL_PATH)) fail("NO MODEL", MODEL_PATH);
    nr_log("[boot] model file %s: %u B\n", MODEL_PATH, (unsigned)g_sd.size());
    if (!read_text(TOOLS_PATH, g_tools, sizeof(g_tools)))
        strcpy(g_tools, g_creature ? kCreatureToolsJson : kToolsJson);
    if (!read_text(LABEL_PATH, g_label, sizeof(g_label))) g_label[0] = 0;
    parse_tool_names();

    // header + directory first (small), then the flash overlay
    Cact probe;
    if (!probe.parse(&g_sd)) fail("BAD MODEL", "not a Needle 3 .cact archive");
    const uint32_t dir_hash = storage_archive_hash(&g_sd, probe.n_recs);
    const uint32_t n_layers = probe.h.num_layers;
    probe.release();
    if (g_part.open("needle")) storage_attach_overlay(&g_rd, &g_part, &g_sd, g_sd.size(), dir_hash);
    if (!g_label[0]) snprintf(g_label, sizeof(g_label), "%u layers", (unsigned)n_layers);

    ui_boot("Reading the tokenizer...", g_label);
    // The tokenizer is the archive's own RAW tensor (the last record).
    {
        Cact c;
        if (!c.parse(&g_rd)) fail("BAD MODEL", "directory");
        CactRec tokrec = c.recs[c.n_recs - 1];
        c.release();
        if (!g_tok.load(&g_rd, tokrec)) fail("BAD MODEL", "tokenizer");
    }
    if (!setup_tokenizer_index(dir_hash)) fail("SD WRITE FAILED", TOKIDX_PATH);

    // Size the KV cache for this tool prefix: prefix + turn + reasoning + call.
    static uint16_t ids[400];
    static char text[2400];
    snprintf(text, sizeof(text), "<|im_start|>user\n<tools>%s</tools>", g_tools);
    int prefix = g_tok.encode(text, ids, 400) + 1;
    if (prefix <= 1) fail("TOOLS TOO LONG", "the tool prefix does not fit");
    // turn text <= 24 tokens + <think>, reasoning, </think>\n<tool_call>, call <= 8
    // the user turn: up to 36 tokens ("A storm is rolling in! I'm starving, exhausted, very
    // curious and sad." is 34); the four-action demo's sentences are shorter
    const uint32_t ctx = (uint32_t)prefix + (g_creature ? 36 : 25) + CONFIG_NEEDLE_THINK_BUDGET + 3 + 8;
    nr_log("[boot] tool prefix %d tokens; context %u positions\n", prefix, (unsigned)ctx);

    report_memory("before model");
    if (!g_model.load(&g_rd, ctx)) fail("OUT OF MEMORY", "loading the model state");
    report_memory("model loaded");
    if (!g_sess.init(&g_model, &g_tok, g_tools, g_name_ptrs, g_ntools, nullptr, 1))
        fail("TOOLS", "grammar");
    uint32_t key = dir_hash ^ storage_content_hash(&g_sd, g_sd.size());
    for (const char* p = g_tools; *p; p++) key = (key ^ (uint8_t)*p) * 16777619u;
    g_snap.configure(SNAP_PATH, key);
    ui_boot("Reading the tool list...", "first boot: a minute or two, then cached");
    uint64_t t0 = nr_micros();
    uint64_t b0 = g_sd.bytes_read;
    if (!g_sess.build_prefix(&g_snap)) fail("PREFIX FAILED", "see the serial log");
    nr_log("[boot] prefix ready in %.1f s (%llu B read)\n", (nr_micros() - t0) / 1e6,
           (unsigned long long)(g_sd.bytes_read - b0));
    report_memory("ready");
    led_set(LED_GREEN);

#if CONFIG_NEEDLE_SELFTEST
    run_bench();
    print_stats();
#endif

    if (g_creature) creature_loop();
    ui_main(g_state, ui_info());
    enum { S_MAIN, S_DECISION, S_DEBUG } screen = S_MAIN;
    bool was_down = false;
    for (;;) {
        poll_serial();
        int x, y;
        bool down = touch_read(&x, &y);
        if (down && !was_down) {
            UiHit h = screen == S_MAIN ? ui_main_hit(x, y)
                      : screen == S_DECISION ? ui_decision_hit(x, y)
                                             : ui_debug_hit(x, y);
            static const int STEPS[] = {10, 30, 50, 70, 90};
            auto cycle = [](int v) {
                for (int s : STEPS) if (s > v) return s;
                return STEPS[0];
            };
            switch (h) {
                case HIT_HUNGER: g_state.hunger = cycle(g_state.hunger); ui_main_value(g_state, 0); break;
                case HIT_ENERGY: g_state.energy = cycle(g_state.energy); ui_main_value(g_state, 1); break;
                case HIT_CURIOSITY: g_state.curiosity = cycle(g_state.curiosity); ui_main_value(g_state, 2); break;
                case HIT_THINK: think_ui(); screen = S_DECISION; break;
                case HIT_AGAIN: case HIT_BACK: ui_main(g_state, ui_info()); screen = S_MAIN; break;
                case HIT_DEBUG: show_debug(); screen = S_DEBUG; break;
                case HIT_MODE: switch_profile(); break;
                default: break;
            }
        }
        was_down = down;
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}
