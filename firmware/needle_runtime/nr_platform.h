// Platform hooks for the Needle runtime: allocation accounting, time, logging.
// The same runtime sources build for the desktop oracle test (host/) and for
// the ESP32 firmware; only this layer differs.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Every runtime allocation goes through here so peak/largest can be reported.
// `tag` names the buffer in the allocation log.
void* nr_alloc(size_t bytes, const char* tag);
void nr_free(void* p);
// For float-only buffers: on the ESP32 prefer spare IRAM (word access is
// native; the rare byte access is emulated), else ordinary heap.
void* nr_alloc_fast32(size_t bytes, const char* tag);

typedef struct {
    size_t current;       // bytes live now
    size_t peak;          // high-water mark of `current`
    size_t largest;       // largest single allocation
    const char* largest_tag;
    uint32_t count;       // allocations made
    size_t iram;          // bytes placed in spare IRAM (ESP32)
} nr_mem_stats;

const nr_mem_stats* nr_mem(void);
void nr_mem_reset_peak(void);

uint64_t nr_micros(void);
void nr_log(const char* fmt, ...);

#ifdef __cplusplus
}
#endif
