# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:59:23 2017

@author: Al
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sys import exit

def read_data(f_in):
    try:
        return pd.read_csv(f_in, index_col = 'subreddit')
    except FileNotFoundError:
        print("File " + f_in + " not found")
        exit(1)

## ****************************************************************************
## * PLOT DATA
## ****************************************************************************
#Make a random array and then make it positive-definite
def plot_data(data, max_k=10):
    # plot the input data
#    plt.figure(); plt.scatter(data[:,0], data[:,1])
#    plt.title('Illustrative data with three distinct clusters')
#    plt.xlabel('x'); plt.ylabel('y')
    
    # plot the curve
    x = range(1,max_k)
    inertia = [run_kmeans(data, i) for i in x]
    plt.figure()
    plt.xticks(x)
    plt.plot(x, inertia, '-o')
    #plt.axvline(x=3, marker='.', color='black', linewidth=0.5)
    plt.title('Change in within-cluster sum of squares with number of clusters')
    plt.xlabel('Number of clusters') 
    plt.ylabel('Total within-cluster sum of squares')
    plt.show()

def run_kmeans(data, k, seed=1):
    km = KMeans(n_clusters=k, random_state=seed, n_jobs=-1).fit(data)
    return km.inertia_
     
## ****************************************************************************
## * RUNNING MODEL
## ****************************************************************************
def main(plotting=True, seed=1):
    """
    Run the Gaussian Mixture Model. Generate data, get initial clusters, run 
    the Expectation Maximisation algorithm until it converges, plot the clusters
    as they evolve.
    
    Parameters:
        - plotting, a boolean to decide whether or not to plot the output (e.g.
        if running the algorithm in more than 2D, plotting won't be possible / 
        relevant)
    """
    
    f_in = "../Data/Output/Experiment2/1511250908_MergedData_logZeroRescale_PCA_Whitened.txt"
    data = read_data(f_in)  
    plot_data(data, max_k=60)
    
    # plot the output
       
if __name__ == '__main__':
    main()