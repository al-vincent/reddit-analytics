# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:40:12 2017

@author: Al
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sys import exit

def read_data(f_in, f_coords):
    
    # get the data to be clustered
    try:
        X = pd.read_csv(f_in, index_col='subreddit')
    except FileNotFoundError:
        print("The data file " + f_in + " was not found")
        exit(1)
        
    # get the 2D visualisation coordinates for the same data
    try:
        coords = pd.read_csv(f_coords, index_col='subreddit')
    except FileNotFoundError:
        print("The coordinate file " + f_coords + " was not found")
        exit(2)
        
    return X, coords

def cluster_points(X, n_clusters):
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    silhouette_values = silhouette_samples(X, cluster_labels)
    return silhouette_avg, silhouette_values, cluster_labels

def plot_silhouette_scores(X, ax1, n_clusters, silhouette_values, cluster_labels,
                           silhouette_avg):
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.3, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
    y_lower = 10
    # iterate through each cluster
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        colour = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=colour, edgecolor=colour, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

def plot_clusters(coords, ax2, cluster_labels, n_clusters, silhouette_values):

    # Setup colours for each point, based on its cluster assignment
    colours = cm.spectral(cluster_labels.astype(float) / n_clusters)
    # set the alpha value for the point, based on its silhouette score (so that
    # points with low silhouette are more transparent)
    alphas = (silhouette_values + 1) / 2
    colours[:,3] = alphas
    # plot the points
    ax2.scatter(coords['x'], coords['y'], marker='.', s=30, lw=0, c=colours, 
                edgecolor='k')

    # set labels for title and axes
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

def main():
    # set the root path to the data file
    path = "../Data/Output/Experiment2/"
    f_in = path+"1511250908_MergedData_logZeroRescale_PCA_Whitened.txt"
    f_coords = path+"TSNE/1511250908_MergedData_logZeroRescale_PCA_Whitened_p50_tSNE.csv"
    
    # get the data to be clustered
    X, coords = read_data(f_in, f_coords)
                                                    
    # set the range of cluster sizes. Will be [MIN_CLUSTERS, MAX_CLUSTERS)
    MIN_CLUSTERS = 10
    MAX_CLUSTERS = 20    
    # iterate through the clusters
    for n_clusters in range(MIN_CLUSTERS,MAX_CLUSTERS):
    
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # cluster the data points and calculate their silhouette scores
        silhouette_avg, silhouette_values, cluster_labels = cluster_points(X, n_clusters)
        
        #print(silhouette_values)
        # plot the silhouette score profile for each cluster    
        plot_silhouette_scores(X, ax1, n_clusters, silhouette_values, 
                               cluster_labels, silhouette_avg)
    
        # 2nd Plot showing the actual clusters formed
        plot_clusters(coords, ax2, cluster_labels, n_clusters, silhouette_values)

        # set the overall plot title    
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
                
        # display the plot
        plt.show()

if __name__ == '__main__':
    main()