/*
 * The routine functions for 2-D or 3-D Barnes-Hut Trees
 *
 * Copyright (c) 2016, Zhirong Yang
 * All rights reserved.
 *
 */

#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <float.h>
#include "barnes_hut.h"

#define MAX(a,b) ( (a) > (b) ? (a) : (b) )
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )
#define MAX_DEPTH 50

QuadTree* createQuadTree(Node *node, double position[], double minPos[], double maxPos[]) {
    QuadTree *tree;
    int d,i;
    tree = (QuadTree*)malloc(sizeof(QuadTree));
    tree->node = node;
    tree->weight = node != NULL ? node->weight : 0.0;
    tree->coef = 0.0;
    tree->childrenLength = PDIM;
    tree->children = (QuadTree**)malloc(sizeof(QuadTree*)*tree->childrenLength);
    for (i=0;i<tree->childrenLength;i++)
        tree->children[i] = NULL;
    tree->childCount = 0;
    for(d=0;d<DIM;d++) {
        tree->position[d] = position[d];
        tree->minPos[d] = minPos[d];
        tree->maxPos[d] = maxPos[d];
    }
    return tree;
}

void destroyQuadTree(QuadTree *tree) {
    int i;
    if (tree==NULL)
        return;
    for (i=0;i<tree->childrenLength;i++) {
        if (tree->children[i]!=NULL)
            destroyQuadTree(tree->children[i]);
    }
    if (tree->node!=NULL)
        free(tree->node);
    if (tree->children!=NULL)
        free(tree->children);
    free(tree);
}

void addNode(QuadTree *tree, Node *newNode, double newPos[], int depth) {
    int d;
    if (tree->node!=NULL) {
        addNode2(tree, tree->node, tree->position, depth);
        tree->node = NULL;
    }
    for(d=0;d<DIM;d++) {
        tree->position[d] = (tree->weight*tree->position[d]+newNode->weight*newPos[d]) / (tree->weight+newNode->weight);
    }
    tree->weight += newNode->weight;
    
    addNode2(tree, newNode, newPos, depth);
}

void addNode2(QuadTree *tree, Node *newNode, double newPos[], int depth) {
    QuadTree **oldChildren;
    int childIndex, d, i, newChildrenLength;
    double newMinPos[DIM], newMaxPos[DIM];
    
    if (depth==MAX_DEPTH) {
        if (tree->childrenLength==tree->childCount) {
            oldChildren = tree->children;
            tree->children = (QuadTree**)malloc(2*sizeof(QuadTree*)*tree->childrenLength);
            newChildrenLength = 2 * tree->childrenLength;
            for (i=0;i<tree->childrenLength;i++)
                tree->children[i] = oldChildren[i];
            for (i=tree->childrenLength;i<newChildrenLength;i++)
                tree->children[i] = NULL;
            tree->childrenLength = newChildrenLength;
            if (oldChildren!=NULL)
                free(oldChildren);
        }
        tree->children[tree->childCount++] = createQuadTree(newNode, newPos, newPos, newPos);
        return;
    }
    
    childIndex = 0;
    for (d=0;d<DIM;d++) {
        if (newPos[d]>(tree->minPos[d]+tree->maxPos[d])/2) {
            childIndex += 1 << d;
        }
    }
    
    if (tree->children[childIndex]==NULL) {
        for(d=0; d<DIM; d++) {
            if ((childIndex & 1<<d)==0) {
                newMinPos[d] = tree->minPos[d];
                newMaxPos[d] = (tree->minPos[d]+tree->maxPos[d])/2;
            } else {
                newMinPos[d] = (tree->minPos[d]+tree->maxPos[d])/2;
                newMaxPos[d] = tree->maxPos[d];
            }
        }
        tree->childCount++;
        tree->children[childIndex] = createQuadTree(newNode, newPos, newMinPos, newMaxPos);
    } else {
        addNode(tree->children[childIndex], newNode, newPos, depth+1);
    }
}

QuadTree* buildQuadTree(double *Y, double *weights, int n) {
    double minPos[DIM], maxPos[DIM], posDiff;
    int d, i;
    QuadTree *tree;
    double dummyPos[DIM];
    Node *node;
    double pos[DIM];
    
    for (d=0;d<DIM;d++) {
        dummyPos[d] = 0.0;
    }
    
    for (d=0;d<DIM;d++) {
        minPos[d] = DBL_MAX;
        maxPos[d] = -DBL_MAX;
    }
    for (i=0;i<n;i++) {
        for (d=0;d<DIM;d++) {
            minPos[d] = MIN(Y[d*n+i], minPos[d]);
            maxPos[d] = MAX(Y[d*n+i], maxPos[d]);
        }
    }
    for (d=0;d<DIM;d++) {
        posDiff = maxPos[d] - minPos[d];
        maxPos[d] += posDiff / 2;
        minPos[d] -= posDiff / 2;
    }
    
    tree = createQuadTree(NULL, dummyPos, minPos, maxPos);
    
    for(i=0;i<n;i++) {
        node = (Node*)malloc(sizeof(Node));
        node->id = i;
        if (weights==NULL)
            node->weight = 1.0;
        else
            node->weight = weights[i];
        for (d=0;d<DIM;d++)
            pos[d] = Y[d*n+i];
        addNode(tree, node, pos, 0);
    }
    
    return tree;
}

double getTreeWidth(QuadTree *tree) {
    double width, widthd;
    int d;
    
    width = 0.0;
    for (d=0;d<DIM;d++) {
        widthd = tree->maxPos[d]-tree->minPos[d];
        if (widthd>width) {
            width = widthd;
        }
    }
    return width;
}

