#include "nr_platform.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "sdkconfig.h"
#else
#include <time.h>
#endif

// A small header in front of each block records its size, so nr_free can
// keep the live count exact without a lookup table.
typedef struct {
    size_t size;
    size_t pad;  // keep the payload 8-byte aligned for doubles/int64
} nr_hdr;

static nr_mem_stats g_mem;

void* nr_alloc(size_t bytes, const char* tag) {
#ifdef ESP_PLATFORM
    nr_hdr* h = (nr_hdr*)heap_caps_malloc(sizeof(nr_hdr) + bytes, MALLOC_CAP_8BIT);
#else
    nr_hdr* h = (nr_hdr*)malloc(sizeof(nr_hdr) + bytes);
#endif
    if (!h) {
        nr_log("nr_alloc: FAILED %u bytes for %s (live %u, peak %u)\n", (unsigned)bytes,
               tag, (unsigned)g_mem.current, (unsigned)g_mem.peak);
        return NULL;
    }
    h->size = bytes;
    h->pad = 0;
    g_mem.current += bytes;
    g_mem.count++;
    if (g_mem.current > g_mem.peak) g_mem.peak = g_mem.current;
    if (bytes > g_mem.largest) {
        g_mem.largest = bytes;
        g_mem.largest_tag = tag;
    }
    void* p = h + 1;
    memset(p, 0, bytes);
#ifndef ESP_PLATFORM
    if (getenv("NR_LOG_ALLOC")) fprintf(stderr, "alloc %7u %s\n", (unsigned)bytes, tag);
#endif
    return p;
}

void* nr_alloc_fast32(size_t bytes, const char* tag) {
#if defined(ESP_PLATFORM)
    // Spare instruction RAM: 32-bit loads/stores only (byte access faults).
    nr_hdr* h = (nr_hdr*)heap_caps_malloc(sizeof(nr_hdr) + bytes, MALLOC_CAP_EXEC | MALLOC_CAP_32BIT);
    if (h) {
        volatile uint32_t* hw = (volatile uint32_t*)h;  // header written as words too
        hw[0] = (uint32_t)bytes;
        hw[1] = 1;  // IRAM: not counted against the DRAM heap figures
        uint32_t* w = (uint32_t*)(h + 1);
        for (size_t i = 0; i < bytes / 4; i++) w[i] = 0;
        g_mem.iram += bytes;
        return h + 1;
    }
#endif
    return nr_alloc(bytes, tag);
}

void nr_free(void* p) {
    if (!p) return;
    nr_hdr* h = ((nr_hdr*)p) - 1;
    if (h->pad == 1) g_mem.iram -= h->size;
    else g_mem.current -= h->size;
#ifdef ESP_PLATFORM
    heap_caps_free(h);
#else
    free(h);
#endif
}

const nr_mem_stats* nr_mem(void) { return &g_mem; }
void nr_mem_reset_peak(void) { g_mem.peak = g_mem.current; }

uint64_t nr_micros(void) {
#ifdef ESP_PLATFORM
    return (uint64_t)esp_timer_get_time();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
#endif
}

void nr_log(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}
