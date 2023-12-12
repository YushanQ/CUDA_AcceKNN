/*
we conduct KNN model on iris dataset (https://www.kaggle.com/datasets/arshid/iris-flower-dataset).

About the dataset
the set contains 150 rows of record. For each row, it has 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species). 
All attributes is presented with numuric value.

Our work
we treat the 4 attributes - Petal Length, Petal Width, Sepal Length, Sepal width as features space, 
and make classification based on their corresponding species via CUDA-KNN model.
We split the dataset by 80% train set and 20% test set, and set k=5(or 3?)
For convenience, we preprocess the dataset, shuffle and split it into 2 files - train.csv and test.csv 
*/

#include "knn.cuh"
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream> 

using namespace std;
vector<vector<float>> Read_file(string filename);
void Write_file(vector<float> &arr);

int main(int argc, char *argv[]) {
    int k = atoi(argv[1]);
    // read file
    vector<vector<float>> data_train; 
    vector<vector<float>> data_test;
    cout << "here0\n";
    data_train = Read_file("train.csv");
    data_test = Read_file("test.csv");
    // cout << "below is test data" << endl;
    // for (auto arr:data_test) {
    //   for (auto e : arr) {
    //     cout << e << ",";
    //   }
    // }
    // cout << "below is train data" << endl;
    // for (auto arr:data_train) {
    //   for (auto e : arr) {
    //     cout << e << ",";
    //   }
    // }
    // cout << endl;
    
    int m_a = data_train.size(), n_a = data_train[0].size();
    int m_b = data_test.size(), n_b = data_test[0].size();

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

    A_catagory = data_train[m_a-1].data();

    for (int i=0; i<m_b-1; ++i) {
        for (int j=0; j<n_b; ++j) {
            B[i*n_b + j] = data_test[i][j];
        }
    }

    for (int i=0; i<n_a; i++) {
        for (int j=0; j<n_b; j++) {
            dist[i*n_b+j] = 0.0f;
            idx[i*n_b+j] = i;
        }
    }
    cout << "here2"<<endl;
    KNN(A, n_a, B, n_b, m_a-1, k, dist, idx);

    // cout << "below is distance matrix" << endl;
    // for (int i = 0; i < n_a; ++i) {
    //     for (int j = 0; j < n_b; ++j) {
    //         cout << dist[i*n_b + j] << " ";
    //     }
    //     cout << endl;
    // }

    // cout << "below is index matrix" << endl;
    // for (int i = 0; i < n_a; ++i) {
    //     for (int j = 0; j < n_b; ++j) {
    //         cout << idx[i*n_b + j] << " ";
    //     }
    //     cout << endl;
    // }

    
    // get the catagory based on idx matrix
    int cnt[n_b][3]{};
    for (int i=0; i<k; i++) {
        for (int j=0; j<n_b; j++) {
            int cat = A_catagory[idx[i*n_b+j]];
            cnt[j][cat]++;
        }
    }

    // return the maximun frequency as result
    vector<float> result;
    for (int i=0; i<n_b; i++) {
        auto curr = cnt[i];
        float predict_cat = max_element(curr, curr+3) - curr;
        result.push_back(predict_cat);
    }

    Write_file(result);
    // for (auto ele : result) {
    //   cout << ele << endl;
    // }

    return 0;
}

vector<vector<float>> Read_file(string filename) {  
    vector<vector<float>> dataset;
    vector<float> petal_len, petal_wid, sepal_len, sepal_wid, species;
    float petal_len_i, petal_wid_i, sepal_len_i, sepal_wid_i;
    string catagory, line;
    ifstream inFile(filename);
    // Close the file
    inFile.close();

    // Reopen the file
    inFile.open(filename);
    int cnt = 0, cat = -1;

    if (inFile.is_open()) {
        cout << "starts reading file " << filename << endl;
        while (getline(inFile, line)) {
           // skip the header
            if (cnt == 0) {
              cnt++;
              continue;
            }; 

            replace(line.begin(), line.end(), '-', '_');
            replace(line.begin(), line.end(), ',', ' ');
            istringstream iss(line);
            cnt++;
            iss >> sepal_len_i >> sepal_wid_i >> petal_len_i >> petal_wid_i >> catagory;
            
            // check iss failure
            if (iss.fail()) {
                cerr << "Error reading line from file: " << filename << endl;
                continue;
            }
            
            sepal_len.push_back(sepal_len_i);
            sepal_wid.push_back(sepal_wid_i);
            petal_len.push_back(petal_len_i);
            petal_wid.push_back(petal_wid_i);
            
            if (catagory == "Iris_setosa") {
                cat = 0;
            } else if (catagory == "Iris_versicolor") {
                cat = 1;
            } else if (catagory == "Iris_virginica") {
                cat = 2;
            } else {
                cat = -1;
            }
            species.push_back(cat);
        }
        inFile.close();
        inFile.clear();
        dataset.push_back(sepal_len);
        dataset.push_back(sepal_wid);
        dataset.push_back(petal_len);
        dataset.push_back(petal_wid);
        dataset.push_back(species);    
    } else {
        cerr << "Error. Please check if file exists";
    }
    cout << "finishes reading file " << filename << endl;
    return dataset;
}

void Write_file(vector<float> &arr) {
    ofstream outFile("result.txt");
    if (outFile.is_open()) {
        cout << "starts outputing file..." << endl;
        // for (auto i=0; i<arr.size(); i++) {
        //     outFile << i << ":" << arr[i] << "\n";
        // }
        for (const auto &e : arr) outFile << e << "\n";
        outFile.close();
    } else {
        cerr << "Error opening the file." << endl;
    }
}

