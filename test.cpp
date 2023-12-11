#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <random>

using namespace std;
int main() 
{ 
    // srand(unsigned(time(0))); 
    vector<vector<int>> arr = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    // using built-in random generator 
    auto rd = default_random_engine {};
    shuffle(begin(arr), end(arr), rd);
  
    // // using randomfunc 
    // random_shuffle(arr.begin(), arr.end(), randomfunc); 
  
    // print out content: 
    cout << "arr contains:" << endl; 
    for (auto v : arr) {
        for (int j=0; j<v.size(); j++)
            cout << v[j] << " ";
        cout << endl;
    }
  
    return 0; 
} 