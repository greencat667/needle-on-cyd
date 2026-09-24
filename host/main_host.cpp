// Desktop build of the ESP32 runtime: the same firmware/needle_runtime
// sources, reading the .cact through stdio. Used to prove the port against
// the reference (tools/check_host.py) and the shipped engine (tools/
// benchmark_desktop.py) before anything runs on the board.
//
//   needle_host --model m.cact --tools tools.min.json --prompt "I'm hungry"
//   needle_host --model m.cact --logits ids.bin out.bin     (oracle check)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../firmware/needle_runtime/inference.h"
#include "../firmware/needle_runtime/nr_platform.h"

struct FileStateIO : StateIO {
    FILE* f;
    bool saving;
    bool io(void* p, uint32_t n) override {
        return (saving ? fwrite(p, 1, n, f) : fread(p, 1, n, f)) == n;
    }
};

struct FileSnapshot : SnapshotStore {
    const char* path;
    FileStateIO fio;
    StateIO* open(bool saving) override {
        fio.f = fopen(path, saving ? "wb" : "rb");
        if (!fio.f) return nullptr;
        fio.saving = saving;
        return &fio;
    }
    void close(StateIO*) override { fclose(fio.f); }
};

struct CaptureSink : LogitSink {
    float* out;
    void consume(uint32_t first, const float* lg, uint32_t n) override {
        memcpy(out + first, lg, n * sizeof(float));
    }
};

static char* slurp(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return nullptr;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* s = (char*)malloc(n + 1);
    fread(s, 1, n, f);
    s[n] = 0;
    fclose(f);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = 0;
    return s;
}

// Tool names in declaration order, pulled from the minified tools JSON.
static int tool_names(const char* json, char names[][32], int cap) {
    int n = 0;
    const char* p = json;
    while (n < cap && (p = strstr(p, "{\"name\":\""))) {
        p += 9;
        int i = 0;
        while (*p && *p != '"' && i < 31) names[n][i++] = *p++;
        names[n++][i] = 0;
    }
    return n;
}

int main(int argc, char** argv) {
    const char *model = nullptr, *tools = nullptr, *prompt = nullptr, *snap = nullptr;
    const char *ids_in = nullptr, *logits_out = nullptr, *tok_text = nullptr;
    NeedleOptions opt;
    int ctx = 320;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model")) model = argv[++i];
        else if (!strcmp(argv[i], "--tools")) tools = argv[++i];
        else if (!strcmp(argv[i], "--prompt")) prompt = argv[++i];
        else if (!strcmp(argv[i], "--snapshot")) snap = argv[++i];
        else if (!strcmp(argv[i], "--think")) opt.think_budget = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--calls")) opt.max_calls = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--allowed-prob")) opt.full_vocab_prob = false;
        else if (!strcmp(argv[i], "--full-prob")) opt.full_vocab_prob = true;
        else if (!strcmp(argv[i], "--trace")) opt.trace = true;
        else if (!strcmp(argv[i], "--ctx")) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--logits")) { ids_in = argv[++i]; logits_out = argv[++i]; }
        else if (!strcmp(argv[i], "--tokenize")) tok_text = argv[++i];
    }
    if (!model) {
        fprintf(stderr, "usage: needle_host --model m.cact [--tools t.json --prompt q | --logits ids.bin out.bin | --tokenize text]\n");
        return 2;
    }
    FileReader fr;
    if (!fr.open(model)) { fprintf(stderr, "cannot open %s\n", model); return 1; }
    Model m;
    if (!m.load(&fr, ctx)) { fprintf(stderr, "model load failed\n"); return 1; }
    Tokenizer tk;
    if (!tk.load(&fr, m.cact.recs[m.cact.n_recs - 1])) { fprintf(stderr, "tokenizer failed\n"); return 1; }
    fprintf(stderr, "loaded %u layers, width %u, %u tensors; runtime RAM %u bytes\n",
            (unsigned)m.L, (unsigned)m.D, (unsigned)m.cact.n_recs, (unsigned)nr_mem()->current);

    if (tok_text) {  // print token ids
        static uint16_t ids[2048];
        int n = tk.encode(tok_text, ids, 2048);
        for (int i = 0; i < n; i++) printf("%u%c", ids[i], i + 1 < n ? ' ' : '\n');
        return n < 0;
    }

    if (ids_in) {  // oracle check: every position's full logits
        FILE* f = fopen(ids_in, "rb");
        static uint16_t ids[1024];
        int n = (int)fread(ids, 2, 1024, f);
        fclose(f);
        float* lg = (float*)malloc((size_t)n * m.out_vocab * sizeof(float));
        float h[1024];
        for (int i = 0; i < n; i++) {
            if (!m.forward(&ids[i], 1, h)) return 1;
            CaptureSink s;
            s.out = lg + (size_t)i * m.out_vocab;
            m.logits(h, &s);
        }
        float conf = NAN;
        m.confidence_logit(&conf);
        FILE* o = fopen(logits_out, "wb");
        fwrite(lg, sizeof(float), (size_t)n * m.out_vocab, o);
        fwrite(&conf, sizeof(float), 1, o);
        fclose(o);
        fprintf(stderr, "%d positions, %llu bytes read in %u reads, peak RAM %u\n", n,
                (unsigned long long)fr.bytes_read, (unsigned)fr.reads, (unsigned)nr_mem()->peak);
        return 0;
    }

    char* tj = slurp(tools);
    if (!tj) { fprintf(stderr, "cannot read tools %s\n", tools); return 1; }
    static char names[8][32];
    const char* np[8];
    int nt = tool_names(tj, names, 8);
    for (int i = 0; i < nt; i++) np[i] = names[i];
    NeedleSession sess;
    if (!sess.init(&m, &tk, tj, np, nt, nullptr, opt.max_calls)) { fprintf(stderr, "session init failed\n"); return 1; }
    FileSnapshot fs;
    fs.path = snap;
    uint64_t t0 = nr_micros();
    if (!sess.build_prefix(snap ? &fs : nullptr)) { fprintf(stderr, "prefix failed\n"); return 1; }
    uint64_t t_prefix = nr_micros() - t0;
    NeedleResult r;
    if (!sess.run(prompt, opt, &r)) { fprintf(stderr, "run failed\n"); return 1; }
    const nr_mem_stats* ms = nr_mem();
    printf("{\"function_calls\":%s,\"reasoning\":\"", r.call_json);
    for (const char* p = r.reasoning; *p; p++) {
        if (*p == '"' || *p == '\\') putchar('\\');
        if (*p == '\n') { fputs("\\n", stdout); continue; }
        putchar(*p);
    }
    printf("\",\"confidence\":%.4f,\"call_prob\":%.4f,\"head_logit\":%.4f,"
           "\"think_tokens\":%d,\"think_truncated\":%s,\"prefix_tokens\":%u,\"turn_tokens\":%u,"
           "\"generated\":%u,\"prefix_ms\":%.1f,\"turn_ms\":%.1f,\"bytes_read\":%llu,\"reads\":%u,"
           "\"peak_ram\":%u,\"largest_alloc\":%u,\"largest_tag\":\"%s\"}\n",
           r.confidence, r.call_prob, r.head_logit, r.reasoning_tokens,
           r.think_truncated ? "true" : "false", (unsigned)r.prefix_tokens,
           (unsigned)r.turn_tokens, (unsigned)r.generated, t_prefix / 1000.0, r.total_us / 1000.0,
           (unsigned long long)r.bytes_read, (unsigned)r.reads, (unsigned)ms->peak,
           (unsigned)ms->largest, ms->largest_tag ? ms->largest_tag : "");
    return 0;
}
