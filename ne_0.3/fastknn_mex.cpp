/*
 * Compile by
 *    mex fastknn_mex.cpp -I/usr/include -L/usr/lib
 *
 * An internal function that compute the KNN indices
 * Usage:
 *    ind = fastknn_mex(X, N, D, K, verbose);
 *
 * Written by Zhirong Yang, adapted from van der Maaten's ball tree 
 * implementation
 */

#include <float.h>
#include <stdlib.h>
#include "vptree.h"
#include "mex.h"

using namespace std;

void fastKNN(double* X, int N, int D, int K, double* ind, int verbose) {
    if (verbose)
        mexPrintf(">>>>>>>>>> N=%d, D=%d\n", N, D);
    
    // Build ball tree on data set
    VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
    vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    for(int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
    tree->create(obj_X);
    if (verbose)
        mexPrintf("ball tree built.\n");
    
    // Loop over all points to find nearest neighbors
    vector<DataPoint> indices;
    vector<double> distances;
    for(int n = 0; n < N; n++) {
        
        if(verbose && (n % 10000 == 0)) mexPrintf(" - point %d of %d\n", n, N);
        
        // Find nearest neighbors
        indices.clear();
        distances.clear();
        tree->search(obj_X[n], K + 1, &indices, &distances);
        
        for(int m = 0; m < K; m++) {
            ind[n*K + m] = (double)(indices[m + 1].index());
        }
    }
    
    // Clean up memory
    obj_X.clear();
    delete tree;
}

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
    double *X, *ind;
    int N, D, K;
    int verbose;
    int i;
    
    X = mxGetPr(prhs[0]);
    if (verbose)
        mexPrintf("X got.\n");
    
    D = mxGetM(prhs[0]);
    N = mxGetN(prhs[0]);
    K = (int)mxGetScalar(prhs[1]);
    verbose = (int)(mxGetScalar(prhs[2]));
    
    if (verbose) {
        mexPrintf("arguments passed: D=%d, N=%d, K=%d, verbose=%d\n", D, N, K, verbose);
    }
    
    plhs[0] = mxCreateDoubleMatrix(N*K, 1, mxREAL);
    ind = mxGetPr(plhs[0]);
    
    if (verbose)
        mexPrintf("result space allocated; before calling fastKNN\n");
    
    fastKNN(X, N, D, K, ind, verbose);
    
    for (i=0;i<N*K;i++)
        ind[i]++;
 
}