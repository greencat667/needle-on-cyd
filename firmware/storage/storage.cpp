#include "storage.h"

#include <string.h>
#include <sys/stat.h>

#include "board.h"
#include "driver/sdmmc_host.h"
#include "driver/sdspi_host.h"
#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_vfs_fat.h"
#include "nr_platform.h"
#include "sdmmc_cmd.h"
#include "sdkconfig.h"

static sdmmc_card_t* g_card = nullptr;

bool storage_mount_sd() {
    esp_vfs_fat_sdmmc_mount_config_t mc = {};
    mc.format_if_mount_failed = false;
    mc.max_files = 4;
    mc.allocation_unit_size = 16 * 1024;
    esp_err_t err;
#if CONFIG_NEEDLE_SD_SDMMC
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = 1;
    err = esp_vfs_fat_sdmmc_mount(SD_MOUNT, &host, &slot, &mc, &g_card);
#else
    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.slot = SPI3_HOST;
    host.max_freq_khz = CONFIG_NEEDLE_SD_MHZ * 1000;
    spi_bus_config_t bus = {};
    bus.mosi_io_num = PIN_SD_MOSI;
    bus.miso_io_num = PIN_SD_MISO;
    bus.sclk_io_num = PIN_SD_SCK;
    bus.quadwp_io_num = -1;
    bus.quadhd_io_num = -1;
    bus.max_transfer_sz = 16 * 1024;
    err = spi_bus_initialize(SPI3_HOST, &bus, SPI_DMA_CH_AUTO);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        nr_log("sd: spi bus init failed (%s)\n", esp_err_to_name(err));
        return false;
    }
    sdspi_device_config_t slot = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot.gpio_cs = (gpio_num_t)PIN_SD_CS;
    slot.host_id = SPI3_HOST;
    err = esp_vfs_fat_sdspi_mount(SD_MOUNT, &host, &slot, &mc, &g_card);
#endif
    if (err != ESP_OK) {
        nr_log("sd: mount failed (%s)\n", esp_err_to_name(err));
        return false;
    }
    nr_log("sd: %s, %llu MB, %u kHz\n", g_card->cid.name,
           (unsigned long long)((uint64_t)g_card->csd.capacity * g_card->csd.sector_size >> 20),
           (unsigned)g_card->max_freq_khz);
    return true;
}

uint64_t storage_sd_free_bytes() {
    uint64_t total = 0, free_b = 0;
    esp_vfs_fat_info(SD_MOUNT, &total, &free_b);
    return free_b;
}

bool PartitionReader::open(const char* label) {
    const esp_partition_t* p =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, label);
    if (!p) return false;
    part_ = p;
    size_ = p->size;
    const void* ptr = nullptr;
    esp_partition_mmap_handle_t h;
    if (esp_partition_mmap(p, 0, p->size, ESP_PARTITION_MMAP_DATA, &ptr, &h) == ESP_OK) {
        map_ = (const uint8_t*)ptr;
        nr_log("flash: `%s` mapped at %p (%u KB)\n", label, ptr, (unsigned)(p->size / 1024));
    } else {
        nr_log("flash: mmap failed, using esp_partition_read\n");
    }
    return true;
}

const uint8_t* PartitionReader::view(uint32_t offset, uint32_t len, uint8_t* buf, uint32_t cap) {
    if (!map_) return TensorReader::view(offset, len, buf, cap);
    if ((uint64_t)offset + len > size_) return nullptr;
    bytes_read += len;
    reads++;
    return map_ + offset;
}

bool PartitionReader::read(uint32_t offset, void* dst, size_t len) {
    uint64_t t0 = nr_micros();
    bool ok;
    if (map_ && (uint64_t)offset + len <= size_) {
        memcpy(dst, map_ + offset, len);
        ok = true;
    } else {
        ok = esp_partition_read((const esp_partition_t*)part_, offset, dst, len) == ESP_OK;
    }
    read_us += nr_micros() - t0;
    bytes_read += len;
    reads++;
    return ok;
}

// Weights can change while the directory stays the same (two fine-tunes of
// one rung), so anything computed from the weights is keyed on their bytes:
// 32 bytes at 256 evenly spaced offsets across the whole file.
uint32_t storage_content_hash(TensorReader* r, uint32_t size) {
    uint32_t h = 2166136261u;
    uint8_t buf[32];
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t off = (uint32_t)((uint64_t)(size - sizeof(buf)) * i / 255);
        if (!r->read(off, buf, sizeof(buf))) return 0;
        for (uint32_t k = 0; k < sizeof(buf); k++) h = (h ^ buf[k]) * 16777619u;
    }
    return h;
}

uint32_t storage_archive_hash(TensorReader* r, uint32_t n_tensors) {
    uint32_t h = 2166136261u;
    uint8_t buf[256];
    uint32_t total = 196 + 28 * 4 + n_tensors * 44;
    for (uint32_t off = 0; off < total; off += sizeof(buf)) {
        uint32_t k = total - off < sizeof(buf) ? total - off : sizeof(buf);
        if (!r->read(off, buf, k)) return 0;
        for (uint32_t i = 0; i < k; i++) h = (h ^ buf[i]) * 16777619u;
    }
    return h;
}

// Overlay header at partition offset 0 (tools/prepare_sd.py):
//   u32 magic 'NDLF', u32 version 1, u32 archive_size, u32 dir_hash,
//   u32 n_ranges, then n x {u32 archive_offset, u32 length, u32 part_offset}
int storage_attach_overlay(OverlayReader* overlay, PartitionReader* part, FileReader* sd,
                           uint32_t archive_size, uint32_t dir_hash) {
    uint32_t h[5];
    if (!part->read(0, h, sizeof(h))) return -1;
    if (h[0] != 0x464C444Eu || h[1] != 1) {
        nr_log("flash: no Needle overlay in the partition\n");
        return 0;
    }
    if (h[2] != archive_size || h[3] != dir_hash) {
        nr_log("flash: overlay belongs to another archive (size %u hash %08x, SD %u %08x); "
               "reading everything from SD\n", (unsigned)h[2], (unsigned)h[3],
               (unsigned)archive_size, (unsigned)dir_hash);
        return 0;
    }
    int n = 0;
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < h[4] && i < (uint32_t)OverlayReader::MAX_RANGES; i++) {
        uint32_t r[3];
        part->read(20 + i * 12, r, sizeof(r));
        // spot-check the copy against the SD bytes at both ends of the range
        const uint32_t k = r[1] < 32 ? r[1] : 32;
        uint8_t a[32], b[32];
        bool same = part->read(r[2], a, k) && sd->read(r[0], b, k) && !memcmp(a, b, k) &&
                    part->read(r[2] + r[1] - k, a, k) && sd->read(r[0] + r[1] - k, b, k) &&
                    !memcmp(a, b, k);
        if (!same) {
            nr_log("flash: range %u does not match the SD archive, skipped\n", (unsigned)i);
            continue;
        }
        overlay->add(r[0], r[1], part, r[2]);
        bytes += r[1];
        n++;
    }
    nr_log("flash: %d archive ranges (%llu B) served from flash\n", n, (unsigned long long)bytes);
    return n;
}

bool FileSnapshot::IO::io(void* p, uint32_t n) {
    return (saving ? fwrite(p, 1, n, f) : fread(p, 1, n, f)) == n;
}

StateIO* FileSnapshot::open(bool saving) {
    io_.f = fopen(path_, saving ? "wb" : "rb");
    if (!io_.f) return nullptr;
    io_.saving = saving;
    uint32_t hdr[2] = {0x50414E53u, key_};  // 'SNAP', key
    if (saving) {
        fwrite(hdr, 4, 2, io_.f);
    } else {
        uint32_t got[2] = {0, 0};
        if (fread(got, 4, 2, io_.f) != 2 || got[0] != hdr[0] || got[1] != key_) {
            fclose(io_.f);
            io_.f = nullptr;
            return nullptr;
        }
    }
    return &io_;
}

void FileSnapshot::close(StateIO*) {
    if (io_.f) fclose(io_.f);
    io_.f = nullptr;
}

// Raw card throughput, below the filesystem: multi-sector reads into an
// aligned internal buffer.
void storage_raw_bench() {
    if (!g_card) return;
    uint8_t* buf = (uint8_t*)heap_caps_malloc(8192, MALLOC_CAP_DMA);
    if (!buf) return;
    uint64_t t0 = nr_micros();
    for (int i = 0; i < 128; i++) sdmmc_read_sectors(g_card, buf, 100000 + i * 16, 16);
    double s = (nr_micros() - t0) / 1e6;
    t0 = nr_micros();
    for (int i = 0; i < 128; i++) sdmmc_read_sectors(g_card, buf, 200000 + i, 1);
    double s1 = (nr_micros() - t0) / 1e6;
    nr_log("[sdraw] 1 MB as 16-sector reads: %.2f s (%.2f MB/s); single sectors: %.2f ms each\n", s,
           1.0 / s, s1 * 1000 / 128);
    heap_caps_free(buf);
}
