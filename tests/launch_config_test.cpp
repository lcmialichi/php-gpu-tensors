#undef NDEBUG
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "../src/backends/cuda/launch_config.cuh"

int main()
{
    assert(cuda_linear_index(0, 256, 0) == 0);
    assert(cuda_linear_index(8388608, 256, 0) == 2147483648ULL);
    assert(cuda_linear_index(16777216, 256, 13) == 4294967309ULL);
    assert(cuda_linear_index(INT_MAX - 1, 1024, 1023) == 2199023254527ULL);
    assert(cuda_grid_stride(INT_MAX, 1024) == 2199023254528ULL);
    assert(cuda_matrix_offset(65536, 65536, 7, 1) == 4294967303ULL);
    assert(cuda_matrix_offset(1, 4294967297ULL, 3, 2147483648ULL) == 10737418241ULL);
    assert(cuda_grid_blocks(INT_MAX).x == INT_MAX);
    assert(cuda_grid_blocks((size_t)INT_MAX + 1).x == 0);
    assert(cuda_grid_blocks(4294967297ULL).x == 0);
    assert(cuda_grid_1d(1).x == 1);
    assert(cuda_grid_1d(256).x == 1);
    assert(cuda_grid_1d(257).x == 2);
    assert(cuda_grid_1d(0).x == 0);
    assert(cuda_grid_1d(1, 0).x == 0);
    assert(cuda_grid_1d(1, 1025).x == 0);
    assert(cuda_grid_1d((size_t)INT_MAX * 256, 256).x == INT_MAX);
    assert(cuda_grid_1d((size_t)INT_MAX * 256 + 1, 256).x == 0);
    assert(cuda_grid_1d(SIZE_MAX, 1).x == 0);
    dim3 grid = cuda_grid_2d(17, 33, 16, 16, 2);
    assert(grid.x == 2 && grid.y == 3 && grid.z == 2);
    assert(cuda_grid_2d(0, 1, 16, 16).x == 0);
    assert(cuda_grid_2d(1, 0, 16, 16).x == 0);
    assert(cuda_grid_2d(1, 1, 0, 16).x == 0);
    assert(cuda_grid_2d(1, 1, 16, 0).x == 0);
    assert(cuda_grid_2d(1, 1, 64, 32).x == 0);
    assert(cuda_grid_2d(1, 1, 16, 16, 65536).x == 0);
    assert(cuda_grid_2d(1, 65536, 1, 1).x == 0);
    assert(cuda_grid_2d(SIZE_MAX, 1, 1, 1).x == 0);
    grid = cuda_grid_2d(INT_MAX, 65535, 1, 1, 65535);
    assert(grid.x == INT_MAX && grid.y == 65535 && grid.z == 65535);
    puts("launch configuration checks passed");
    return 0;
}
