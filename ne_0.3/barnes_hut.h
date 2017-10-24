/*
 * The routine functions for 2-D or 3-D Barnes-Hut Trees
 *
 * Copyright (c) 2014, Zhirong Yang (Aalto University)
 * All rights reserved.
 *
 */

#ifndef BARNES_HUT
#define BARNES_HUT

#ifndef DIM
#define DIM 2
#endif

#define PDIM (1<<DIM)

typedef struct tagNode {
    int id;
    double weight;
} Node;

typedef struct tagQuadTree{
    Node *node;
    struct tagQuadTree **children;
    int childrenLength;
    int childCount;
    double position[DIM];
    double weight;
    double coef; /*reserved for application use*/
    double minPos[DIM];
    double maxPos[DIM];
} QuadTree;

/* externally callable functions */
QuadTree* buildQuadTree(double *Y, double *weights, int n); /* weights==NULL would use 1.0 for each node weight*/
void destroyQuadTree(QuadTree *tree);
double getTreeWidth(QuadTree *tree);

/* internal functions */
QuadTree* createQuadTree(Node *node, double position[], double minPos[], double maxPos[]);
void addNode(QuadTree *tree, Node *newNode, double newPos[], int depth);
void addNode2(QuadTree *tree, Node *newNode, double newPos[], int depth);

#endif
