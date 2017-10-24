/*
 * Compute the repulsive part of the LinLog objective and gradient
 *
 * Copyright (c) 2014, Zhirong Yang (Aalto University)
 * All rights reserved.
 *
 */

#include "mex.h"
#include <math.h>
#include "barnes_hut.h"

void getRepulsiveObjGradi(int id, double *pos, QuadTree* tree, double theta, int nArgOut, double *pqsum, double *grad) {
    double dist2, diff, q;
    double tmp, tmp2;
    int i,d;
    
    if (tree==NULL) return;
    if (tree->node!=NULL && tree->node->id==id) return;
    
    dist2 = 0.0;
    for (d=0;d<DIM;d++) {
        diff = pos[d]-tree->position[d];
        dist2 += diff*diff;
    }

    tmp2 = theta*getTreeWidth(tree);
    if (tree->childCount>0 && dist2<tmp2*tmp2) {
        for(i=0;i<tree->childrenLength;i++) {
            if (tree->children[i]!=NULL) {
                getRepulsiveObjGradi(id, pos, tree->children[i], theta, nArgOut, pqsum, grad);
            }
        }
    } else {
        (*pqsum) += log(dist2) * tree->weight;

        if (nArgOut>1) {
              tmp = tree->weight / dist2;
              for (d=0;d<DIM;d++) {
                   grad[d] += (tree->position[d] - pos[d]) * tmp;
              }
        }
    }
}

void getRepulsiveObjGrad(double *Y, int n, double eps, double theta, double lambda, int nArgOut, double *pobj, double *grad) {
    int i,d;
    double qsum, pos[DIM], gradi[DIM];
    QuadTree *tree;
    
    tree = buildQuadTree(Y,NULL,n);
    
    qsum = 0.0;
    for (i=0;i<n;i++) {
        for(d=0;d<DIM;d++) {
            gradi[d] = 0.0;
            pos[d] = Y[d*n+i];
        }
        getRepulsiveObjGradi(i, pos, tree, theta, nArgOut, &qsum, gradi);
        if (nArgOut>1) {
            for(d=0;d<DIM;d++)
                grad[d*n+i] = 2 * lambda * gradi[d];
        }
    }
        
    (*pobj) = - 0.5 * lambda * qsum;
    
    destroyQuadTree(tree);
}

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    double *Y, eps, *pobj, *grad, theta, lambda;
    int n, nArgOut;
    
    Y = mxGetPr(prhs[0]);
    theta = mxGetScalar(prhs[1]);
    lambda = mxGetScalar(prhs[2]);
    nArgOut = (int)mxGetScalar(prhs[3]);
    
    n = mxGetM(prhs[0]);
    eps = mxGetEps();

    plhs[0] = mxCreateDoubleScalar(0.0);
    pobj = mxGetPr(plhs[0]);
    if (nArgOut>1) {
        plhs[1] = mxCreateDoubleMatrix(n,DIM,mxREAL);
        grad = mxGetPr(plhs[1]);
    }
    else
        grad = NULL;

    getRepulsiveObjGrad(Y, n, eps, theta, lambda, nArgOut, pobj, grad);
}
