#include "knn.cuh"
#include <stdio.h>

#define BLOCK_DIM 16

__global__ void calDistance_kernel(float* adj, float* tg, unsigned int numAdj, unsigned int dim) {
    __shared__ float shrd_mem[BLOCK_DIM][BLOCK_DIM];
    
}
