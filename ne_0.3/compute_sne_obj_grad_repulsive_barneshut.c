/*
 * Compute the repulsive part of objective and gradient of Stochastic 
 * Neighbor Embedding (SNE)
 *
 * Copyright (c) 2014, Zhirong Yang (Aalto University)
 * All rights reserved.
 *
 */

#include "mex.h"
#include <math.h>
#include "barnes_hut.h"

void getSumqi(int id, double *pos, QuadTree *tree, double theta, double *psumq) {
    double dist2, diff, q;
    double tmp2;
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
                getSumqi(id, pos, tree->children[i], theta, psumq);
            }
        }
    } else {
        q = exp(-dist2);
        (*psumq) += q * tree->weight;
    };    
}

double aggregateCoeffcient(QuadTree* tree, double *coef) {
    int i;
    
    if (tree->node==NULL) {
        for (i=0;i<tree->childrenLength;i++) {
            if (tree->children[i]!=NULL)
                tree->coef += aggregateCoeffcient(tree->children[i], coef);
        }
    } else {
        tree->coef = coef[tree->node->id];
    }
    return tree->coef;
}

void getSNERepulsiveObjGradi(int id, double *pos, QuadTree* tree, double *coef, double* sumq, double theta, int nArgOut, double *pobj, double *grad) {
    double dist2, dist, diff, decoef, q;
    double tmp2;
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
                getSNERepulsiveObjGradi(id, pos, tree->children[i], coef, sumq, theta, nArgOut, pobj, grad);
            }
        }
    } else {
        q = exp(-dist2);
        
        if (nArgOut>1) {
            decoef = (tree->weight * coef[id] + tree->coef) * q;
            for (d=0;d<DIM;d++) {
                grad[d] += (pos[d]-tree->position[d]) * decoef;
            }
        }
    }
}

void getSNERepulsiveObjGrad(double *Y, int n, double eps, double theta, int nArgOut, double *pobj, double *grad) {
    double pos[DIM], gradi[DIM], d2, diff;
    double *coef;
    mwIndex i, j, d;
    QuadTree *tree;
    double *sumq;
    
    tree = buildQuadTree(Y,NULL,n);
    
    (*pobj) = 0.0;
    for (i=0;i<n;i++)
        for(d=0;d<DIM;d++)
            gradi[d] = 0.0;
    
    /* Round 1: calculate sum q */
    sumq = (double*)malloc(sizeof(double)*n);
    for (i=0;i<n;i++) {
        sumq[i] = 0.0;
        for(d=0;d<DIM;d++) {
            pos[d] = Y[d*n+i];
        }
        getSumqi(i, pos, tree, theta, sumq+i);
    }
    
    (*pobj) = 0;
    for (i=0;i<n;i++)
        (*pobj) += log(sumq[i]+eps);

    if (nArgOut>1) {
        /* aggregate coefficents in quad tree */
        coef = (double*)malloc(sizeof(double)*n);
        for (i=0;i<n;i++)
            coef[i] = -1.0 / (sumq[i]+eps);
        aggregateCoeffcient(tree, coef);
    }
    else
        coef = NULL;
    
    /* Round 2: calculate obj and gradient */
    for (i=0;i<n;i++) {
        for(d=0;d<DIM;d++) {
            gradi[d] = 0.0;
            pos[d] = Y[d*n+i];
        }
        getSNERepulsiveObjGradi(i, pos, tree, coef, sumq, theta, nArgOut, pobj, gradi);
        
        if (nArgOut>1) {
            for(d=0;d<DIM;d++) {
                grad[d*n+i] += gradi[d];
                grad[d*n+i] *= 2;
            }
        }
    }
    
    destroyQuadTree(tree);
    
    free(sumq);
    if (nArgOut>1) {
        free(coef);
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    double *Y, eps, *pobj, *grad, theta;
    int n, nArgOut;
    
    Y = mxGetPr(prhs[0]); /* note that Y is a vector */
    theta = mxGetScalar(prhs[1]);
    nArgOut = (int)mxGetScalar(prhs[2]);
    
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
    

    getSNERepulsiveObjGrad(Y, n, eps, theta, nArgOut, pobj, grad);
}
