// Pure-read memory bandwidth probe: many threads streaming large arrays.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <chrono>

int main(int argc, char** argv) {
    size_t bytes = (argc > 1 ? atoll(argv[1]) : 1024) * (1ull << 20);
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    int maxth = omp_get_max_threads();
    unsigned long long* a = (unsigned long long*)aligned_alloc(64, bytes);
    if (!a) { printf("alloc fail\n"); return 1; }
    memset(a, 1, bytes);  // populate
    printf("array %zu MiB, max threads %d\n", bytes >> 20, maxth);
    for (int th = 1; th <= maxth; th *= 2) {
        omp_set_num_threads(th);
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            unsigned long long s = 0;
            #pragma omp parallel for reduction(+:s) schedule(static)
            for (size_t i = 0; i < bytes / 8; ++i) s += a[i];
            auto t1 = std::chrono::high_resolution_clock::now();
            if (s == 0) printf("");  // keep
            double dt = std::chrono::duration<double>(t1 - t0).count();
            if (dt < best) best = dt;
        }
        printf("threads %2d: %7.1f GB/s\n", th, (double)bytes / best / 1e9);
    }
    return 0;
}
