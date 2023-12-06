#include "knn.cuh"
#include <iostream>
#include <cuda.h>
#include <stdio.h>

using namespace std;

int main(int argc, char *argv[]) {

    int adj_test[20] = {1,2,3,5,7,9,11,13,15,17, 2,3,4,6,8,10,12,14,16,18};
    int tg_test[8] = {4,8,12,16, 5,9,13,17};
    int y_test[10] = {0, 1, 0, 1, 1, 1, 0, 0, 1, 1};

    float *A, *B, *dist, *idx;
    cudaMallocManaged((void **)&A, sizeof(float) * 20);
  	cudaMallocManaged((void **)&B, sizeof(float) * 8);
    cudaMallocManaged((void **)&dist, sizeof(float) * 40);
    cudaMallocManaged((void **)&idx, sizeof(float) * 40);
    A = adj_test;
    B = tg_test;

    for (unsigned int i=0; i<10; i++) {
        for (unsigned int j=0; j<4; j++) {
            dist[i*4+j] = 0.0f;
            idx[i*4+j] = i;
        }
    }

    //KNN(float* A, unsigned int nA, float *B, unsigned int nB, unsigned int dim, unsigned int k, float* dist, int* idx)
    KNN(A, nA=10, B, nB=4, dim=2, k=3, dist, idx);

    // get the catagory based on idx matrix
    int cnt[4][3]{};
    int k=3;
    for (unsigned int i=0; i<k; i++) {
        for (unsigned int j=0; j<4; j++) {
            int cat = idx[i*4+j];
            cnt[j][cat]++;
        }
    }

    int ans[4]{};
    int catagory_num = 2;
    for (int i=0; i<4; i++) {
        int rec = MIN_INT;
        for (int j=0; j<catagory_num; j++) {
            if (cnt[i][j]) {
                rec = max(rec, cnt[i][j]);
            }
        }
        ans[i] = rec;
    }

    cout << ans << endl;
}