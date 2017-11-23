# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:25:17 2017

@author: Al
"""

import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
from sklearn.manifold import SpectralEmbedding, MDS, TSNE
from sys import exit

def read_data(f_in):
    try:
        return pd.read_csv(f_in, index_col = 'subreddit')
    except FileNotFoundError:
        print("File " + f_in + " not found")
        exit(1)

def run_spectral_embedding(data, n_neighbours=1, seed=1):
    
    coords= SpectralEmbedding(random_state=seed, n_neighbors=n_neighbours,
                              eigen_solver="arpack", n_jobs=-1).fit_transform(data)
    
    return pd.DataFrame({"x":coords[:,0], "y":coords[:,1], 
                         "subreddit":data.index.values}).set_index('subreddit')

def run_multi_dimensional_scaling(data, seed=1):
        
    coords = MDS(n_jobs=-1, random_state=seed).fit_transform(data)
    
    return pd.DataFrame({"subreddit":data.index.values, 
                  "x":coords[:,0], "y":coords[:,1]}).set_index('subreddit')

def run_tsne(data, perplexity=50, seed=1):
        
    coords = TSNE(random_state=seed, perplexity=perplexity).fit_transform(data)
    
    return pd.DataFrame({"subreddit":data.index.values, 
                  "x":coords[:,0], "y":coords[:,1]}).set_index('subreddit')
    
def plot_data(coords, method):#, df, proxi_dists, d_min, d_max):
        """
        Plot the proxigram. Create a scattergraph using t-SNE points and 
        subreddit names, then plot arrows between each point and its knn. 
        Colour arrows based on distance (in feature space, *not* t-SNE space).
        
        Parameters:
            - tsne, dataframe of t-SNE (x,y) coords for each subreddit
            - df, dataframe of original subreddit feature values
            - proxi-dists, dict-of-dicts holding knn, distances per subreddit
            - d_min, minimum distance in feature space
            - d_max, maximum distance in feature space
            - plot_lines, boolean, indicates whether to plot proxi lines. If 
                False, a normal scatterplot is plotted (MUCH quicker)
            - plot_names, boolean, indicates whether to include subreddit names 
                on plot (takes longer, but helpful).                                                                     
        """
        # turn off interactive mode (only shows plots if plt.show() is used)
        plt.ioff()
        
        names = coords.index.values
        
        # create a scatterplot showing subreddits on t-SNE coordinates
        plt.figure(figsize=(13, 10))
        ax = plt.axes()    
        ax.scatter(x=coords['x'],y=coords['y'],s=2)#,color='black'
                    
        # add subreddit names as labels to points
        if names is not None:
            for i, txt in enumerate(names):            
                ax.text(coords.iloc[i][0], coords.iloc[i][1], txt, fontsize=2)
    
        plt.title("Subreddit visualisation using " + method)
        plt.show()

def main():
    
    f_in = "../Data/Output/Experiment2/1511250908_MergedData_logZeroRescale_PCA_Whitened.txt"
    df = read_data(f_in)    
    
    spectral = run_spectral_embedding(df)
    print(spectral.head())
    plot_data(spectral, "spectral embedding")
    
    mds = run_multi_dimensional_scaling(df)
    print(mds.head())
    plot_data(mds, "multi-dimensional scaling")
    
    tsne = run_tsne(df)
    print(tsne.head())
    plot_data(tsne, "t-SNE")

if __name__ == '__main__':
    main()