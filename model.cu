'''
we conduct KNN model on iris dataset (https://www.kaggle.com/datasets/arshid/iris-flower-dataset).

About the dataset
the set contains 150 rows of record. For each row, it has 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species). 
All attributes is presented with numuric value.

Our work
we treat the 4 attributes - Petal Length, Petal Width, Sepal Length, Sepal width as features space, 
and make classification based on their corresponding species via CUDA-KNN model.
'''

#include "knn.cuh"
#include <iostream>
#include <cuda.h>
#include <stdio.h>

using namespace std;

int main(int argc, char *argv[]) {


    
}

