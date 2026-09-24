// Calls the shipped engine's needle_embed() to read its confidence-head probe
// pool for a text: the black-box check on the port's probe-pool maths.
#include <stdio.h>
#include <stdlib.h>
#include "../../engines/macos-arm64/needle.h"
static unsigned char* slurp(const char* p, unsigned long long* n) {
    FILE* f = fopen(p, "rb"); fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char* b = malloc(*n); fread(b, 1, *n, f); fclose(f); return b;
}
int main(int argc, char** argv) {
    unsigned long long n; unsigned char* w = slurp(argv[1], &n);
    if (needle_load(w, n) < 0) { printf("load: %s\n", needle_last_error()); return 1; }
    int r = needle_init("", argv[2], NULL);
    if (r < 0) { printf("init: %s\n", needle_last_error()); return 1; }
    int dim = needle_embed(argv[3], NULL, 0);
    float* out = malloc(dim * 4);
    needle_embed(argv[3], out, dim);
    fwrite(out, 4, dim, stdout);
    fprintf(stderr, "prefix %d, dim %d\n", r, dim);
    return 0;
}
