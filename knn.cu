#include "knn.cuh"
#include <stdio.h>
#include <iostream>
#define BLOCK_DIM 16
using namespace std;

__device__ void swap(float *dist, int *idx, int i, int j, int w) {
    float temp_dist = dist[i*w];
    dist[i*w] = dist[j*w];
    dist[j*w] = temp_dist;

    int temp_idx = idx[i*w];
    idx[i*w] = idx[j*w];
    idx[j*w] = temp_idx;
}


__device__ int partition(float *dist_col, int *idx_col, int start, int end, int w) {

    float pivot = dist_col[end*w];
    int i = start - 1;

    for (int j = start; j < end; j++) {
        if (dist_col[j*w] <= pivot) {
            i++;
            swap(dist_col, idx_col, i, j, w);
        }
    }
    swap(dist_col, idx_col, i+1, end, w);
    
    // printf("Partitioning: start = %d, end = %d, pivot index = %d\n", start, end, i+1);
    
    return i+1;
}


__device__ void quicksort(float *dist_col, int *idx_col, int start, int end, int w) {
    // printf("Quicksort called: start = %d, end = %d\n", start, end);
    
    if (start >= end) return;
    int pivot_idx = partition(dist_col, idx_col, start, end, w);
    
    // Recursively sort the two parts
    quicksort(dist_col, idx_col, start, pivot_idx - 1, w);
    quicksort(dist_col, idx_col, pivot_idx + 1, end, w);
}


__global__ void calDistance_kernel(float *A, float *B, float *dist, unsigned int nA, unsigned int nB, unsigned int dim) {
    __shared__ float sMem_a[BLOCK_DIM][BLOCK_DIM];
    __shared__ float sMem_b[BLOCK_DIM][BLOCK_DIM];

    // rename threadidx
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float sum = 0.f; 

    // setting start and end range
    // checking matrix A by y, and matrix B by x, because we will consider cross multiply
    // __shared__ int Bg_a = BLOCK_DIM * blockIdx.y;
    // __shared__ int Sp_a = BLOCK_DIM * nA;           // moving begin pointer down the matrix
    // __shared__ int Ed_a = Bg_a + nA * (dim-1);      // end at the last element in the matrix
    // __shared__ int Bg_b = BLOCK_DIM * blockIdx.x;
    // __shared__ int Ed_b = BLOCK_DIM * nB;

    __shared__ int Bg_a, Sp_a, Ed_a, Bg_b, Sp_b;
    Bg_a = BLOCK_DIM * blockIdx.y;
    Sp_a = BLOCK_DIM * nA; 
    Ed_a = Bg_a + nA * (dim-1);
    Bg_b = BLOCK_DIM * blockIdx.x;
    Sp_b = BLOCK_DIM * nB;

    for (int i=Bg_a, j=Bg_b; i<Ed_a; i+=Sp_a, j+=Sp_b) {
        // add element value to share memory, if index i is out-of-bound, fill shared memory with 0
        // if (tx==0 && ty ==0) {
        //   printf("dim is %d, ", dim);
        //   printf("i/nA + ty is %d, ", i/nA + ty);
        //   printf("Bg_a + tx is %d, ", Bg_a + tx);
        //   printf("A value is %f", A[i + ty*nA + tx]);
        //   printf("here");
        // }
        if (i/nA + ty < dim) {
            if (Bg_a + tx < nA) {
                sMem_a[ty][tx] = A[i + ty*nA + tx];
            } else {
                sMem_a[ty][tx] = 0;
            }

            // if (tx==0 && ty ==0) {
            //   printf("%.2f\n", sMem_a[ty][tx]);
            //   // should be 1 here
            // }

            // if (tx==0 && ty ==1) {
            //   printf("%.2f\n", sMem_a[ty][tx]);
            //   // should be 2 here
            // }

            if (Bg_b + tx < nB) {
                sMem_b[ty][tx] = B[j + ty*nB + tx];
            } else {
                sMem_b[ty][tx] = 0;
            }

            // if (tx==0 && ty ==0) {
            //   printf("%.2f\n", sMem_b[ty][tx]);
            //   // should be 4 here
            // }

            // if (tx==0 && ty ==1) {
            //   printf("%.2f\n", sMem_b[ty][tx]);
            //   // should be 5 here
            // }
        } else {
            sMem_a[ty][tx] = 0;
            sMem_b[ty][tx] = 0;
        }
        __syncthreads();

        // calculate distant follows format: dist = (x_A - x_B)^2
        if (Bg_b + tx < nB && Bg_a + ty < nA) {
            for (int k=0; k<BLOCK_DIM; k++) {
                sum += pow(sMem_a[k][ty] - sMem_b[k][tx], 2);
                // if (tx==0 && ty ==0) {
                //   printf("%.2f\n", sum);
                // }
            }
        }
        __syncthreads();
    }

    // store dist in dist matrix
    if (Bg_b + tx < nB && Bg_a + ty < nA) {
        // if (tx==0 && ty ==0) {
        //   printf("%.2f\n", sum);
        // }
        dist[(Bg_a+ty)*nB + Bg_b+tx] = sum;
        // if (tx==3 && ty == 9) {
        //   printf("dist: %.2f\n", dist[0]);
        // }   
    }
}


// may had to rewrite sort
__global__ void sortDistance_kernel(float *dist, int *idx, unsigned int w, unsigned int h, unsigned int k) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < w) {
        float* dist_col = dist + tid;
        int* idx_col = idx + tid;
        // find the kth smallest element in dist and return their corresponding idx with quicksort
        // quicksort(dist_col, idx_col, 0, h-1, int w);
        //
        // some unexpected problem happened on quicksort, we change it to bubble sort, the answer is correct, with some sacrifices to runtime.
        int i, j;
        bool swapped;
        for (i = 0; i < h - 1; i++) {
            swapped = false;
            for (j = 0; j < h - i - 1; j++) {
                if (dist_col[j*w] > dist_col[(j+1)*w]) {
                    // swap(arr[j], arr[j + 1]);
                    swap(dist_col, idx_col, j, j+1, w);
                    swapped = true;
                }
            }

            if (swapped == false)
                break;
        }
        
    }
}


void KNN(float *A, 
         unsigned int nA, 
         float *B, 
         unsigned int nB, 
         unsigned int dim, 
         unsigned int k, 
         float *dist, 
         int *idx) {
    
    dim3 blk_dim_cal((nB+BLOCK_DIM-1)/BLOCK_DIM, (nA+BLOCK_DIM-1)/BLOCK_DIM, 1);
    dim3 thrd_per_blk_cal(BLOCK_DIM, BLOCK_DIM, 1);

    int thrd_per_blk_sort = BLOCK_DIM*BLOCK_DIM;
    int blk_dim_sort = (nB + thrd_per_blk_sort -1) / thrd_per_blk_sort;

    calDistance_kernel<<<blk_dim_cal, thrd_per_blk_cal>>>(A, B, dist, nA, nB, dim);
    sortDistance_kernel<<<blk_dim_sort, thrd_per_blk_sort>>>(dist, idx, nB, nA, k);

    cudaDeviceSynchronize();

    // cout << "below is distance matrix" << endl;
    // for (int i = 0; i < nA; ++i) {
    //     for (int j = 0; j < nB; ++j) {
    //         cout << dist[i*nB + j] << " ";
    //     }
    //     cout << endl;
    // }

    // cout << "below is index matrix" << endl;
    // for (int i = 0; i < nA; ++i) {
    //     for (int j = 0; j < nB; ++j) {
    //         cout << idx[i*nB + j] << " ";
    //     }
    //     cout << endl;
    // }
    
}
