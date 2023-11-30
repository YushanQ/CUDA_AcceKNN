// Author: Yushan Qin and Pin Chun Lu
// ref https://github.com/vincentfpgarcia/kNN-CUDA/blob/master/code/knncuda.cu
// ref https://github.com/unlimblue/KNN_CUDA/blob/master/knn_cuda/csrc/cuda/knn.cu

#ifndef KNN_CUH
#define KNN_CUH

// Computes the distence between target point with all adjacent points provided
// Each thread should compute one distance between the target and an adjacent point
//
// @param A     adjacent points
// @param B     target points
// @param dist  distance matrix
// @nA          number of adjacent points
// @nB          number of target points
// @dim         dimension of points
__global__ void calDistance_kernel(float* A, float* B, float* dist, unsigned int nA, unsigned int nB, unsigned int dim); 


// Sort distances and select K-nearest points
// 
// @param dist  distance matrix
// @param idx   index matrix
// @param w     width of distance matrix
// @param h     height of distance matrix
// @param k     k nearest points will be considered
__global__ void sortDistance_kernel(float* dist, int* idx, unsigned int w, unsigned int h, unsigned int k);


// Makes one call to calDistance_kernel and sortDistance_kernel with threads_per_block threads per block.
void KNN(float* adj, unsigned int numAdj, float *tg, unsigned int dim, unsigned int k, float* dist, int* idx);

#endif

