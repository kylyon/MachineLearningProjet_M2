#include "library.h"
#define DLLEXPORT extern "C" __declspec(dllexport)

#include <iostream>

float* transposeMatrix(float* mat, int row, int col)
{
    float* transposeM = new float[row * col];
    for (int i = 0; i < col; ++i) {
        for (int j = 0; j < row; ++j) {
            transposeM[i + j * col] = mat[j + i * col];
        }
    }

    return transposeM;
}

float* multiplyMatrix(float* matA, float* matB, int rowA, int colA, int colB)
{
    float* multM = new float[rowA * colB];
    for (int i = 0; i < rowA; ++i) {
        for (int j = 0; j < colB; ++j) {
            for (int k = 0; k < colA; ++k) {
                multM[j + i * rowA] = matA[k + i * rowA] * matB[j + k * colA];
            }
        }
    }

    return multM;
}

DLLEXPORT void saveML(char* filen,float* W, int size, int yCol)
{
    string filename = string(filen);

    ofstream fichier(filename, ios::out | ios::trunc);

    fichier << size << endl;
    fichier << yCol << endl;

    for (int j = 0; j < size * yCol; j++) {
        fichier << W[j] << " ";
    }
    fichier << endl;

    cout << "Linear Model Saved" << endl;
}

DLLEXPORT void saveRBF(char* filen,float* W, float* uks, int size, int yCol, int xUKS, int yUKS)
{
    string filename = string(filen);

    ofstream fichier(filename, ios::out | ios::trunc);

    fichier << size << endl;
    fichier << yCol << endl;
    fichier << xUKS << endl;
    fichier << yUKS << endl;

    for (int j = 0; j < xUKS * yUKS; j++) {
        fichier << uks[j] << " ";
    }
    fichier << endl;

    for (int j = 0; j < size * yCol; j++) {
        fichier << W[j] << " ";
    }
    fichier << endl;

    cout << "Linear Model Saved" << endl;
}

DLLEXPORT int* loadInfoML(char* filen)
{
    string filename = string(filen);

    ifstream file(filename, ios::in);

    int size, yCol;

    file >> size >> yCol;

    int* res = new int[2]{size, yCol};

    return res;
}

DLLEXPORT int* loadInfoRBF(char* filen)
{
    string filename = string(filen);

    ifstream file(filename, ios::in);

    int size, yCol, xUKS, yUKS;

    file >> size >> yCol >> xUKS >> yUKS;

    int* res = new int[4]{size, yCol, xUKS, yUKS};

    return res;
}

DLLEXPORT float* loadML(char* filen)
{
    string filename = string(filen);

    ifstream file(filename, ios::in);

    int size;

    file >> size;

    int yCol;

    file >> yCol;

    float* W = new float[size * yCol];

    for (int i = 0; i < yCol; i++) {
        for (int j = 0; j < size; j++) {
            file >> W[i * size + j];
        }
    }

    return W;
}

DLLEXPORT float* loadRBF(char* filen)
{
    string filename = string(filen);

    ifstream file(filename, ios::in);

    int size;

    file >> size;

    int yCol;

    file >> yCol;

    int xUKS;

    file >> xUKS;

    int yUKS;

    file >> yUKS;

    float* W = new float[size * yCol + xUKS * yUKS];

    for (int i = 0; i < yUKS; i++) {
        for (int j = 0; j < xUKS; j++) {
            file >> W[i * xUKS + j];
        }
    }

    for (int i = 0; i < yCol; i++) {
        for (int j = 0; j < size; j++) {
            file >> W[(xUKS * yUKS) + i * size + j];
        }
    }

    return W;
}

DLLEXPORT float* LinearModelTrain(int nb_rep, float step, float* X, float* Y, int xRow, int xCol, int yCol, int yIndex, bool is_classification)
{
    srand( (unsigned)time(NULL) );

    int xColSize = xCol+1;

    //Init W
    float* W = new float[xColSize];
    for (int i = 0; i < xColSize; i++) {
        W[i] = ((float(rand()) / float(RAND_MAX)) * (1 - -1)) + -1;
    }

    if (!is_classification)
    {

        MatrixXf mX(xRow, xCol);
        MatrixXf mY(xRow, yCol);

        for (int i = 0; i < xRow; i++) {
            for (int j = 0; j < xCol; j++) {
                mX(i,j) = X[j + i * xCol];
            }
        }

        for (int i = 0; i < xRow; i++) {
            for (int j = 0; j < yCol; j++) {
                mY(i,j) = Y[j + i * xCol];
            }
        }


        MatrixXf transposeX = mX.transpose();

        MatrixXf multTX = transposeX * mX;

        MatrixXf invX = multTX.inverse();

        MatrixXf resX  = invX * transposeX;

        MatrixXf res = resX * mY;

        W = new float[res.size()];

        for (int i = 0; i < res.rows(); i++) {
            for (int j = 0; j < res.cols(); j++) {
                W[j + i * res.cols()] = res(i,j);
            }
        }
        return W;
        //float* multTransXX = multiplyMatrix(transposeX, X, xCol, xRow, xCol);
    }



    for (int i = 0; i < nb_rep; i++) {

        //k
        int k = rand() % xRow;

        //Yk
        float yk = Y[k * yCol + yIndex];

        //Xk
        float* xk = new float [xColSize];
        xk[0] = 1.0;
        for (int j = 1; j < xColSize; j++) {
            xk[j] = X[(j-1) + k * xCol];
        }

        //gXk
        float gXk = -1.0;
        float temp = 0;
        for (int j = 0; j < xColSize; j++) {
            temp += W[j] * xk[j];
        }
        if (temp > 0)
        {
            gXk = 1.0;
        }

        //W
        for (int j = 0; j < xColSize; j++) {
            W[j] += step * (yk - gXk) * xk[j];
        }
    }
    //cout << "Linear Model Training done" << endl;
    return W;

}

PMC::PMC(int* npl, int L)
{
    srand( (unsigned)time(NULL) );

    this->W = vector<float*>();
    this->d = vector<int>();
    for (int i = 0; i < L; i++) {
        this->d.push_back(npl[i]);
    }

    this->L = L;

    this->X = vector<float*>();
    this->deltas = vector<float*>();

    /*Initialisation des W*/
    for (int l = 0; l < L; l++) {
        if (l > 0)
        {
            W.push_back(new float[(this->d[l-1] + 1) * (this->d[l] + 1)]);
            for (int i = 0; i < (d[l-1] + 1); i++) {
                for (int j = 0; j < (d[l] + 1); j++) {
                    W[l][j + (i * (d[l-1] + 1))] = j == 0 ? 0.0 : ((float(rand()) / float(RAND_MAX)) * (1 - -1)) + -1;
                }
            }
        }
        else
        {
            W.push_back(new float[1]);
        }
    }

    /*Initialisation des X et des deltas*/
    for (int l = 0; l < L; l++) {
        X.push_back(new float[(d[l] + 1)]);
        deltas.push_back(new float[(d[l] + 1)]);
        for (int j = 0; j < (d[l] + 1); j++) {
            X[l][j] = j == 0 ? 1.0 : 0.0;
            deltas[l][j] = 0.0;
        }
    }

    cout << "PMC created" << endl;
}

PMC::PMC(string filename)
{
    ifstream file(filename, ios::in);

    file >> this->L;

    this->d = vector<int>();
    for (int i = 0; i < this->L; i++) {
        int dL;
        file >> dL;
        this->d.push_back(dL);
    }

    this->W = vector<float*>();
    this->X = vector<float*>();
    this->deltas = vector<float*>();

    /*Initialisation des W*/
    for (int l = 0; l < L; l++) {
        if (l > 0)
        {
            W.push_back(new float[(this->d[l-1] + 1) * (this->d[l] + 1)]);
            for (int i = 0; i < (d[l-1] + 1); i++) {
                for (int j = 0; j < (d[l] + 1); j++) {
                    float Wf;
                    file >> Wf;
                    W[l][j + (i * (d[l-1] + 1))] = Wf;
                }
            }
        }
        else
        {
            W.push_back(new float[1]);
        }
    }

    /*Initialisation des X et des deltas*/
    for (int l = 0; l < L; l++) {
        X.push_back(new float[(d[l] + 1)]);
        for (int j = 0; j < (d[l] + 1); j++) {
            float Xf;
            file >> Xf;
            X[l][j] = Xf;
        }
    }

    for (int l = 0; l < L; l++) {
        deltas.push_back(new float[(d[l] + 1)]);
        for (int j = 0; j < (d[l] + 1); j++) {
            float deltasf;
            file >> deltasf;
            deltas[l][j] = deltasf;
        }
    }

    file.close();

    cout << "PMC created" << endl;
}

void PMC::SavePMC(string filename)
{
    ofstream fichier(filename, ios::out | ios::trunc);

    fichier << this->L << endl;
    for (int i = 0; i < this->L; i++) {
        fichier << this->d[i] << " ";
    }
    fichier << endl;


    for (int l = 0; l < this->L; l++) {
        if (l > 0)
        {
            for (int i = 0; i < (this->d[l-1] + 1); i++) {
                for (int j = 0; j < (this->d[l] + 1); j++) {
                    fichier << this->W[l][j + (i * (this->d[l-1] + 1))]<< " ";
                }
            }
        }
    }
    fichier << endl;

    /*Initialisation des X et des deltas*/
    for (int l = 0; l < this->L; l++) {
        for (int j = 0; j < (this->d[l] + 1); j++) {
            fichier << this->X[l][j] << " ";
        }
    }

    fichier << endl;

    for (int l = 0; l < this->L; l++) {
        for (int j = 0; j < (this->d[l] + 1); j++) {
            fichier << this->deltas[l][j] << " ";
        }
    }

    fichier.close();

    cout << "PMC Saved" << endl;
}

void PMC::Train(int nb_rep, float step, float* X, float* Y, int xRow, int xCol, int yCol, bool is_classification)
{

    srand( (unsigned)time(NULL) );

    int xColSize = xCol+1;

    int k;
    float * Xk;
    float * Yk;

    for (int it = 0; it < nb_rep; it++) {
        //k
        k = rand() % xRow;

        //Yk
        Yk = new float [yCol];
        for (int j = 0; j < yCol; j++) {
            Yk[j] = Y[k * yCol + j];
        }

        //Xk
        Xk = new float [xCol];
        for (int j = 0; j < xCol; j++) {
            Xk[j] = X[(j) + k * xCol];
        }

        Propagate(Xk, is_classification);
        for (int j = 1; j < this->d[this->L - 1] + 1; j++) {
            this->deltas[this->L - 1][j] = this->X[this->L - 1][j] - Yk[j - 1];
            if (is_classification)
                this->deltas[this->L - 1][j] = this->deltas[this->L - 1][j] * (1 - ::powf(this->X[this->L - 1][j],2) );
        }

        for (int l = this->L-1; l >= 1; l--) {
            for (int i = 1; i < this->d[l-1] + 1; i++) {
                float total = 0.0;
                for (int j = 1; j < this->d[l] + 1; j++) {
                    total += this->W[l][j + i * (this->d[l-1] + 1)] * this->deltas[l][j];
                    this->deltas[l - 1][i] = (1 - ::powf(this->X[l - 1][i], 2)) * total;
                }
            }
        }

        for (int l = 1; l < this->L; l++) {
            for (int i = 0; i < this->d[l-1] + 1; i++) {
                for (int j = 1; j < this->d[l] + 1; j++) {
                    this->W[l][j + i * (this->d[l-1] + 1)] += -step * this->X[l-1][i] * this->deltas[l][j];
                }
            }
        }
    }
    cout << "PMC Training done" << endl;
}

float* PMC::Predict(float* X, bool isClassification)
{
    this->Propagate(X, isClassification);

    float* res = new float[this->d[this->L -1]];
    for (int i = 0; i < this->d[this->L -1]; i++) {
        res[i] = this->X[this->L - 1][i+1];
    }
    return res;
}

void PMC::Propagate(float* X, bool isClassification) {
    //cout << "d[0] + 1 = " << this->d[0] << endl;

    for (int j = 1; j < this->d[0] + 1; j++) {
        this->X[0][j] = X[j - 1];
    }

    for (int l = 1; l < this->L; l++) {
        for (int j = 1; j < this->d[l] + 1; j++) {
            float total = 0;
            for (int i = 0; i < this->d[l - 1] + 1; i++) {
                total += this->W[l][j + i * (this->d[l-1] + 1)] * this->X[l - 1][i];
            }

            this->X[l][j] = total;
            if (isClassification or l < (this->L - 1))
            {
                this->X[l][j] = std::tanhf(total);
            }
        }
    }
}


DLLEXPORT PMC* createPMC(int* npl, int L, PMC* pmc)
{
    return new PMC(npl, L);
}

DLLEXPORT void checkPMC(PMC* pmc)
{
    for (int i = 0; i < pmc->L; ++i) {
        cout << pmc->d[i] << endl;
    }

}

DLLEXPORT void trainPMC(PMC* pmc, int nb_rep, float step, float* X, float* Y, int xRow, int xCol, int yCol, bool is_classification)
{
    pmc->Train(nb_rep, step, X, Y, xRow, xCol, yCol, is_classification);

}

DLLEXPORT float* predictPMC(PMC* pmc, float* X, bool isClassification)
{
    return pmc->Predict(X, isClassification);
}

DLLEXPORT void savePMC(PMC* pmc)
{
    //cout << "test" << endl;
    pmc->SavePMC("test.txt");
}

DLLEXPORT PMC* createPMCFromFile(char* file)
{
    //cout << "test" << endl;
    return new PMC(string(file));
}

void PMC::free() {
    vector<int> d;
    vector<float*> W;
    vector<float*> X;
    vector<float*> deltas;
}

DLLEXPORT void freeMemory(PMC* pmc)
{
    //pmc->free();
    ::free(pmc);
}

bool isPartitionStable(vector<vector<float>> uk, vector<vector<float>> ukOld)
{
    if (uk.size() != ukOld.size())
    {
        return false;
    }
    for (int i = 0; i < uk.size(); i++) {
        for (int j = 0; j < uk[i].size() ;j++) {
            if (uk[i][j] != ukOld[i][j])
            {
                return false;
            }
        }
    }
    return true;
}

DLLEXPORT float* kMeansAlgo(float* X, int k, int xRow, int xCol)
{
    srand( (unsigned)time(NULL) );
    vector<vector<vector<float>>> Ck = vector<vector<vector<float>>>();

    vector<int> indexK = vector<int>();
    for (int i = 0; i < xRow; i++) {
        indexK.push_back( ((float(rand()) / float(RAND_MAX)) * (k - 0)));
    }

    //Init Cks
    for (int i = 0; i < k; i++) {
        Ck.push_back(vector<vector<float>>());
    }

    //Populate Cks
    for (int i = 0; i < xRow; i++) {
        Ck[indexK[i]].push_back(vector<float>());
        for (int j = 0; j < xCol; j++) {
            Ck[indexK[i]].back().push_back(X[i * xCol + j]);
        }
    }

    int index = 0;
    vector<vector<float>> uk = vector<vector<float>>();
    vector<vector<float>> ukOld = vector<vector<float>>();
    vector<vector<vector<float>>> Ckp1;
    bool hasChanged = false;
    while (true) {
        uk.clear();

        //Init centre µk
        for (int i = 0; i < k; i++) {
            uk.push_back(vector<float>());
            for (int j = 0; j < xCol; j++) {
                uk[i].push_back(0.0f);
            }
        }

        for (int i = 0; i < k; i++) {
            for (int j = 0; j < Ck[i].size() ;j++) {
                for (int l = 0; l < xCol; l++) {
                    uk[i][l] += Ck[i][j][l] / Ck[i].size();
                }
            }
        }

        Ckp1 = vector<vector<vector<float>>>();

        //Init Ckp1s
        for (int i = 0; i < k; i++) {
            Ckp1.push_back(vector<vector<float>>());
        }

        //Populate Ckp1s
        for (int i = 0; i < Ck.size(); i++) {
            for (int m = 0; m < Ck[i].size(); m++) {
                float minDistance = numeric_limits<float>::max();
                int minIndex = 0;
                for (int l = 0; l < k; l++) {
                    float temp = 0;
                    for (int j = 0; j < xCol; j++) {
                        temp += pow(Ck[i][m][j] - uk[l][j], 2);
                    }
                    temp = ::abs(::sqrt(temp));
                    if (temp < minDistance)
                    {
                        minDistance = temp;
                        minIndex = l;
                    }
                }

                if(i != minIndex && !hasChanged)
                {
                    hasChanged = true;
                }
                Ckp1[minIndex].push_back(vector<float>());
                for (int j = 0; j < xCol; j++) {
                    Ckp1[minIndex].back().push_back(Ck[i][m][j]);
                }

            }
        }

       /* for (int i = 0; i < k; i++) {
            cout << "[";
            for (int j = 0; j < xCol; j++) {
                cout << " " << uk[i][j];
            }
            cout << " ]" << endl;
        }

        cout << endl;*/

        if (!hasChanged)
        {
            break;
        }

        hasChanged = false;
        Ck = Ckp1;
        ukOld = uk;
        index++;

    }

    float* res = new float[k * xCol];
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < xCol; j++) {
            res[i * xCol + j] = uk[i][j];
        }
    }

    // Tableau k * xCol
    return res;
}

