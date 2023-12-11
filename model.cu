'''
we conduct KNN model on iris dataset (https://www.kaggle.com/datasets/arshid/iris-flower-dataset).

About the dataset
the set contains 150 rows of record. For each row, it has 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species). 
All attributes is presented with numuric value.

Our work
we treat the 4 attributes - Petal Length, Petal Width, Sepal Length, Sepal width as features space, 
and make classification based on their corresponding species via CUDA-KNN model.
We split the dataset by 80% train set and 20% test set, and set k=5(or 3?)
For convenience, we preprocess the dataset, shuffle and split it into 2 files - train.csv and test.csv 
'''

#include "knn.cuh"
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <algorithm>
#include <random>

using namespace std;

int main(int argc, char *argv[]) {
    int k = atoi(argv[1]);
    // read file
    vector<vector<float>> data_train, data_test;
    Read_file(data_train, true);
    Read_file(data_test, false);
    int m_a = data_train.size(), n_a = data_train[0].size();
    int m_b = data_test.size(), n_b = data_test.size();

    // generate A and B matrix
    int sz_a = (m_a-1) * n_a;
    int sz_b = (m_b-1) * n_b;

    float *A, *A_catagory, *B, *dist;
    int *idx;
    cudaMallocManaged((void **)&A, sizeof(float) * sz_a);
  	cudaMallocManaged((void **)&B, sizeof(float) * sz_b);
    cudaMallocManaged((void **)&dist, sizeof(float) * n_a * n_b);
    cudaMallocManaged((void **)&idx, sizeof(float) * n_a * n_b);

    for (int i=0; i<m_a-1; ++i) {
        for (int j=0; j<n_a; ++j) {
            A[i*n_a + j] = data_train[i][j];
        }
    }

    for (int j=0; j<n_a; ++j) {
        A_catagory[j] = data_train[m_a-1][j];
    }

    for (int i=0; i<m_b-1; ++i) {
        for (int j=0; j<n_b; ++j) {
            B[i*n_b + j] = data_test[i][j];
        }
    }

    for (unsigned int i=0; i<n_a; i++) {
        for (unsigned int j=0; j<n_b; j++) {
            dist[i*n_b+j] = 0.0f;
            idx[i*n_b+j] = i;
        }
    }

    KNN(A, n_a, B, n_b, m_a-1, k, dist, idx);

    // get the catagory based on idx matrix
    int cnt[n_b][3]{};
    for (int i=0; i<k; i++) {
        for (int j=0; j<n_b; j++) {
            int cat = A_catagory[idx[i*n_b+j]];
            cnt[j][cat]++;
        }
    }





}

vector<vector<float>> Read_file(vector<vector<float>> &dataset, bool train) {
    string filename = train ? "train.csv" : "test.csv";
    
    vector<float> petal_len, petal_wid, sepal_len, sepal_wid, species;
    float petal_len_i, petal_wid_i, sepal_len_i, sepal_wid_i;
    string catagory, line;
    ifstream myfile(filename);
    int cnt = 0, cat = -1;

    if (myfile.is_open()) {
        cout << "starts reading file..." << endl;
        while (getline(myfile, line)) {
            replace(line.begin(), line.end(), '-', '_');
            replace(line.begin(), line.end(), ',', ' ');
            istringstream iss(line);
            cnt++;

            iss >> sepal_len_i >> sepal_wid_i >> petal_len_i >> petal_wid_i >> catagory;
            sepal_len.push_back(sepal_len_i);
            sepal_wid.push_back(sepal_wid_i);
            petal_len.push_back(petal_len_i);
            petal_wid.push_back(petal_wid_i);
            
            switch(catagory) {
                case "Iris_setosa":
                    cat = 0; 
                    break;
                case "Iris-versicolor":
                    cat = 1; 
                    break;
                case "Iris-virginica":
                    cat = 2; 
                    break;
            }
            species.push_back(cat);
        }
        dataset.push_back(sepal_len);
        dataset.push_back(sepal_wid);
        dataset.push_back(petal_len);
        dataset.push_back(petal_wid);
        dataset.push_back(species);  
    } else {
        cerr << "Error. Please check if file exists";
    }
}

