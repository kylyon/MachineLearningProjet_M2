#ifndef CPPLIB_LIBRARY_H
#define CPPLIB_LIBRARY_H


#include <iostream>
#include "limits"
#include <math.h>
#include <time.h>
#include "string"
#include <vector>
#include "Eigen/Eigen"
#include "fstream"

using Eigen::MatrixXf;

using namespace std;

class PMC
{
    public:
        vector<vector<vector<float>>> W;
        vector<float*> X;
        vector<float*> deltas;
        vector<int> d;
        int L;

        PMC(int* npl, int L);
        PMC(string filename);
        void SavePMC(string filename);
        void Train(int nb_rep, float step, float* X, float* Y, int xRow, int xCol, int yCol, bool isClassification);
        float* Predict(float* X, bool isClassification);

        void free();

    private:
        void Propagate(float* X, bool isClassification);
};


#endif //CPPLIB_LIBRARY_H
