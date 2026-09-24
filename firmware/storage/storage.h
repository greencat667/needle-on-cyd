// Storage for the ESP32: the SD card (model archive, prefix snapshot,
// tokenizer index) and the `needle` flash partition (a verbatim copy of the
// archive's hottest byte ranges, laid down by tools/prepare_sd.py).
#pragma once
#include <stdint.h>
#include <stdio.h>

#include "cact_reader.h"
#include "inference.h"

#define SD_MOUNT "/sdcard"
#define SD_DIR SD_MOUNT "/needle"

bool storage_mount_sd();
void storage_raw_bench();
uint64_t storage_sd_free_bytes();

// esp_partition_read over the `needle` data partition.
class PartitionReader : public TensorReader {
public:
    bool open(const char* label);
    bool read(uint32_t offset, void* dst, size_t len) override;
    // Memory-mapped: the bytes are served in place through the flash cache.
    const uint8_t* view(uint32_t offset, uint32_t len, uint8_t* buf, uint32_t cap) override;
    uint32_t size() const { return size_; }
    bool mapped() const { return map_ != nullptr; }
private:
    const void* part_ = nullptr;
    uint32_t size_ = 0;
    const uint8_t* map_ = nullptr;
};

// Validate the partition's overlay header against the SD archive and route
// its ranges through `overlay`. Returns the number of ranges attached.
int storage_attach_overlay(OverlayReader* overlay, PartitionReader* part, FileReader* sd,
                           uint32_t archive_size, uint32_t dir_hash);

// FNV-1a over the archive's header + directory: ties side files to one model.
uint32_t storage_archive_hash(TensorReader* r, uint32_t n_tensors);
// Sampled fingerprint of the weight bytes (keys the prefix snapshot).
uint32_t storage_content_hash(TensorReader* r, uint32_t size);

// Snapshot of the static prefix, on SD, keyed by the archive and tool set.
class FileSnapshot : public SnapshotStore {
public:
    void configure(const char* path, uint32_t key) { path_ = path; key_ = key; }
    StateIO* open(bool saving) override;
    void close(StateIO* io) override;
private:
    struct IO : StateIO {
        FILE* f = nullptr;
        bool saving = false;
        bool io(void* p, uint32_t n) override;
    } io_;
    const char* path_ = nullptr;
    uint32_t key_ = 0;
};
