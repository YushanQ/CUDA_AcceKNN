#include "knn.cuh"
#include <stdio.h>

#define BLOCK_DIM 16

__global__ void calDistance_kernel(float* A, float* B, float* dist, unsigned int nA, unsigned int nB, unsigned int dim) {
    __shared__ float sMem_a[BLOCK_DIM][BLOCK_DIM];
    __shared__ float sMem_b[BLOCK_DIM][BLOCK_DIM];

    // rename threadidx
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float sum = 0.f; 

    // setting start and end range
    // checking matrix A by y, and matrix B by x, because we will consider cross multiply
    __shared__ int Bg_a = BLOCK_DIM * blockIdx.y;
    __shared__ int Sp_a = BLOCK_DIM * nA;           // moving begin pointer down the matrix
    __shared__ int Ed_a = Bg_a + nA * (dim-1);      // end at the last element in the matrix
    __shared__ int Bg_b = BLOCK_DIM * blockIdx.x;
    __shared__ int Ed_b = BLOCK_DIM * nB;

    for (int i=Bg_a, j=Bg_b; i<Ed_a; i+=Sp_a, j+=Sp_b) {
        // add element value to share memory, if index i is out-of-bound, fill shared memory with 0
        if (i/nA + ty < dim) {
            if (Bg_a + tx < nA) {
                sMem_a[ty][tx] = A[i + ty*nA + tx];
            } else {
                sMem_a[ty][tx] = 0;
            }

            if (Bg_b + tx < nB) {
                sMem_b[ty][tx] = B[j + ty*nA + tx];
            } else {
                sMem_b[ty][tx] = 0;
            }
        } else {
            sMem_a[ty][tx] = 0;
            sMem_b[ty][tx] = 0;
        }
        __syncthreads();

        // calculate distant follows format: dist = (x_A - x_B)^2
        if (Bg_b + tx < nB && Bg_a + ty < nA) {
            for (int k=0; k<BLOCK_DIM; k++) {
                sum += pow(sMem_a[k][ty] - sMem_b[k][tx], 2);
            }
        }
        __syncthreads();
    }

    // store dist in dist matrix
    if (Bg_b + tx < nB && Bg_a + ty < nA) {
        dist[(Bg_a+ty)*nB + Bg_b+tx] = sum;
    }
}
