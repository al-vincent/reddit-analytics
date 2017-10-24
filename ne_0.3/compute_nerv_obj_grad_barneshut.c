/*
 * Compute the objective and gradient of Neighbor Retrieval Visualizer (NeRV)
 *
 * Copyright (c) 2014, Zhirong Yang (Aalto University)
 * All rights reserved.
 *
 */

#include "mex.h"
#include <math.h>
#include "barnes_hut.h"

void getSumqi(int id, double *pos, QuadTree *tree, double farness_factor, double *psumq) {
    double dist2, dist, diff, q;
    int i,d;
    
    if (tree==NULL) return;
    if (tree->node!=NULL && tree->node->id==id) return;
    
    dist2 = 0.0;
    for (d=0;d<DIM;d++) {
        diff = pos[d]-tree->position[d];
        dist2 += diff*diff;
    }
    dist = sqrt(dist2);
    
    if (tree->childCount>0 && dist<farness_factor*getTreeWidth(tree)) {
        for(i=0;i<tree->childrenLength;i++) {
            if (tree->children[i]!=NULL) {
                getSumqi(id, pos, tree->children[i], farness_factor, psumq);
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

void getNeRVObjGradDensei(int id, double *pos, QuadTree* tree, double lambda, double epsilon, double eps, double logepsilon, double *coef, double* sumq, double farness_factor, int nArgOut, double *pobj, double *grad) {
    double dist2, dist, diff, decoef, q, Q;
    int i,d;
    
    if (tree==NULL) return;
    if (tree->node!=NULL && tree->node->id==id) return;
    
    dist2 = 0.0;
    for (d=0;d<DIM;d++) {
        diff = pos[d]-tree->position[d];
        dist2 += diff*diff;
    }
    dist = sqrt(dist2);
    
    if (tree->childCount>0 && dist<farness_factor*getTreeWidth(tree)) {
        for(i=0;i<tree->childrenLength;i++) {
            if (tree->children[i]!=NULL) {
                getNeRVObjGradDensei(id, pos, tree->children[i], lambda, epsilon, eps, logepsilon, coef, sumq, farness_factor, nArgOut, pobj, grad);
            }
        }
    } else {
        q = exp(-dist2);
        Q = q / (sumq[id]+eps);
        
        (*pobj) -= (lambda*epsilon*log(Q+eps) + (1-lambda)*logepsilon*Q) * tree->weight;
        
        if (nArgOut>1) {
            decoef = (tree->weight * coef[id] + tree->coef) * q;
            for (d=0;d<DIM;d++) {
                grad[d] += (pos[d]-tree->position[d]) * decoef;
            }
        }
    }
}

void getNeRVObjGrad(double *Y, mwIndex *irs, mwIndex *jcs, double *Pval, int n, double lambda, double epsilon, double eps, double farness_factor, int nArgOut, double *pobj, double *grad) {
    double pos[DIM], gradi[DIM], d2, diff;
    double q, P, Q, logP, logQ;
    double tmp, spcoef, constant, logepsilon, epsilonlogepsilon;
    mwIndex ind, i, j, d, row, col;
    mwIndex starting_row_index, stopping_row_index, current_row_index;
    QuadTree *tree;
    double *sumq, *innerSum, *coef;
/*    double sumq[10000], innerSum[10000], coef[10000];*/
    double gradmax;

    logepsilon = log(epsilon + eps);
    epsilonlogepsilon = epsilon * logepsilon; 
    
    tree = buildQuadTree(Y,NULL,n);
    
    /* Round 1: calculate sum q */
    sumq = (double*)malloc(sizeof(double)*n);
    if (sumq==NULL)
        mexPrintf("!!!!!! malloc sumq failed !!!!!!\n");
    for (i=0;i<n;i++) {
        sumq[i] = 0.0;
        for(d=0;d<DIM;d++) {
            pos[d] = Y[d*n+i];
        }
        getSumqi(i, pos, tree, farness_factor, sumq+i);
    }
    
    /* Round 2: calculate sparse part, recall constant and inner loop variables */
    if (nArgOut>1) {
        innerSum = (double*)malloc(sizeof(double)*n);
        if (innerSum==NULL)
            mexPrintf("!!!!!! malloc innerSum failed !!!!!!\n");
        for (i=0;i<n;i++) {
            innerSum[i] = 0.0;
            for (d=0;d<DIM;d++)
                grad[d*n+i] = 0.0;
        }
    }
    else
        innerSum = NULL;
    
    (*pobj) = 0.0;
    ind = 0;
    for (col=0; col<n; col++) {
        starting_row_index = jcs[col];
        stopping_row_index = jcs[col+1];
        if (starting_row_index == stopping_row_index)
            continue;
        else {
            for (current_row_index = starting_row_index;
            current_row_index < stopping_row_index;
            current_row_index++)  {
                row = irs[ind];
                P = Pval[ind];
                
                if (row!=col) {
                    d2 = 0.0;
                    for (d=0; d<DIM; d++) {
                        diff = Y[d*n+row]-Y[d*n+col];
                        d2 += diff * diff;
                    }
                    
                    q = exp(-d2);
                    Q = q / (sumq[col]+eps);
                    logQ = log(Q+eps);
                    logP = log(P+eps);
                    
                    (*pobj) += lambda*(epsilon-P)*logQ
                               +(1-lambda)*Q*(logepsilon+logQ-logP);
                    
                    constant += P*logP - epsilonlogepsilon;
                    
                    if (nArgOut>1) {
                        tmp = logP -logQ - 1 - logepsilon;
                        spcoef = lambda*(P-epsilon)+(1-lambda)*(Q*tmp);
                        for (d=0; d<DIM; d++) {
                            grad[d*n+col] += spcoef * (Y[d*n+col] - Y[d*n+row]);
                            grad[d*n+row] += spcoef * (Y[d*n+row] - Y[d*n+col]);
                        }
                        
                        innerSum[col] -= Q*tmp;
                    }
                }
                ind++;
            }
        }
    }
    
    (*pobj) += (constant + epsilonlogepsilon*n*n) * lambda;

    if (nArgOut>1) {
        /* aggregate coefficents in quad tree */
        coef = (double*)malloc(sizeof(double)*n);
        if (coef==NULL)
            mexPrintf("!!!!!! malloc coef failed !!!!!!\n");
        for (i=0;i<n;i++)
            coef[i] = (-lambda+(1-lambda)*innerSum[i]) / (sumq[i]+eps);
        aggregateCoeffcient(tree, coef);
    }
    else
        coef = NULL;
            
    /* Round 3: calculate dense part */
    for (i=0;i<n;i++) {
        for(d=0;d<DIM;d++) {
            gradi[d] = 0.0;
            pos[d] = Y[d*n+i];
        }
        getNeRVObjGradDensei(i, pos, tree, lambda, epsilon, eps, logepsilon, coef, sumq, farness_factor, nArgOut, pobj, gradi);
        
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
        free(innerSum);
        free(coef);
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    double *Y, *Pval, lambda, epsilon, eps, *pobj, *grad, farness_factor;
    int n, nArgOut;
    mwIndex *irs, *jcs;
    
    Y = mxGetPr(prhs[0]); /* note that Y is a vector */
    Pval = mxGetPr(prhs[1]); /* a vector that contains the nonzeros values */
    lambda = mxGetScalar(prhs[2]);
    epsilon = mxGetScalar(prhs[3]);
    farness_factor = mxGetScalar(prhs[4]);
    nArgOut = (int)mxGetScalar(prhs[5]);
    
    n = mxGetM(prhs[1]);
    irs = mxGetIr(prhs[1]);
    jcs = mxGetJc(prhs[1]);
    eps = mxGetEps();
    
    plhs[0] = mxCreateDoubleScalar(0.0);
    pobj = mxGetPr(plhs[0]);
    if (nArgOut>1) {
        plhs[1] = mxCreateDoubleMatrix(n*DIM,1,mxREAL);
        grad = mxGetPr(plhs[1]);
    }
    else
        grad = NULL;
    
    getNeRVObjGrad(Y, irs, jcs, Pval, n, lambda, epsilon, eps, farness_factor, nArgOut, pobj, grad);
}
