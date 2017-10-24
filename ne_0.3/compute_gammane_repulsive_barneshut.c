#include "mex.h"
#include <math.h>
#include "barnes_hut.h"

#define KERNEL_GAUSSIAN 0
#define KERNEL_CAUCHY 1
#define KERNEL_INVSQD 2

#define GAMMA_EPS 1e-10

void getRepulsiveObjGradi(int id, double *pos, QuadTree* tree, double gamma, double theta, int kerneltype, int nArgOut, double *pqsum, double *grad) {
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
                getRepulsiveObjGradi(id, pos, tree->children[i], gamma, theta, kerneltype, nArgOut, pqsum, grad);
            }
        }
    } else {
        switch (kerneltype) {
            case KERNEL_GAUSSIAN:
                q = exp(-dist2);
                break;
            case KERNEL_CAUCHY:
                q = 1/(1+dist2);
                break;
            case KERNEL_INVSQD:
                q = 1/dist2;
                break;                
        }
        
        if (fabs(gamma+1)<GAMMA_EPS) {
            if (kerneltype==KERNEL_GAUSSIAN)
                tmp = - tree->weight * dist2;
            else
                tmp = tree->weight * log(q);
/*            printf("pos=(%f,%f), tree->position=(%f,%f), dist2=%f, q=%f, log(q)=%f, tree->weight=%f, tmp=%f\n", pos[0], pos[1], tree->position[0], tree->position[1], dist2, q, log(q), tree->weight, tmp);*/
        }
        else if (fabs(gamma)<GAMMA_EPS) {
            tmp = tree->weight * q;
        }
        else if (fabs(gamma-1)<GAMMA_EPS) {
            tmp = tree->weight * q * q;
        }
        else {
            tmp = tree->weight * pow(q, 1+gamma);
        }
        
        (*pqsum) += tmp;

        if (nArgOut>1) {
            if (gamma==-1)
                tmp = tree->weight;
            
            switch (kerneltype) {
                case KERNEL_CAUCHY:
                case KERNEL_INVSQD:
                    tmp = tmp * q;
                    break;
            }
              for (d=0;d<DIM;d++) {
                   grad[d] += (tree->position[d] - pos[d]) * tmp;
              }
        }
    }
}

void getRepulsiveObjGrad(double *Y, int n, double eps, double gamma, double theta, int kerneltype, int nArgOut, double *pobj, double *grad) {
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
        getRepulsiveObjGradi(i, pos, tree, gamma, theta, kerneltype, nArgOut, &qsum, gradi);
/*    printf("i=%d, gamma=%f, kerneltype=%d, qsum=%f\n", i, gamma, kerneltype, qsum);*/
        if (nArgOut>1) {
            for(d=0;d<DIM;d++)
                grad[d*n+i] = 4 * gradi[d];
        }
    }
    
    if (nArgOut>1) {
        for (i=0;i<n;i++) {
            for(d=0;d<DIM;d++) {
                if (gamma==-1)
                    grad[d*n+i] /= n*(n-1);
                else
                    grad[d*n+i] /= qsum;
            }
        }
    }
    
    if (gamma==-1)
        (*pobj) = qsum/(n*(n-1));
    else
        (*pobj) = log(qsum+eps)/(1+gamma);
    
    destroyQuadTree(tree);
}

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    double *Y, eps, *pobj, *grad, theta, gamma;
    int n, nArgOut, kerneltype;
    
    Y = mxGetPr(prhs[0]);
    gamma = mxGetScalar(prhs[1]);
    theta = mxGetScalar(prhs[2]);
    kerneltype = mxGetScalar(prhs[3]);
    nArgOut = (int)mxGetScalar(prhs[4]);
    
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

    getRepulsiveObjGrad(Y, n, eps, gamma, theta, kerneltype, nArgOut, pobj, grad);
}
