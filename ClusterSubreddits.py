# -*- coding: utf-8 -*-
"""****************************************************************************
Created on Tue Aug 22 13:46:23 2017

@author:        A. VINCENT
Description:    Program to apply three clustering methods to data extracted 
                the media platform Reddit. The three methods used are:
                    - Agglomerative hierarchical Clustering;
                    - Dirichlet Process Gaussian Mixture Models;
                    - Density-Based Spatial Clustering of Applications with 
                      Noise (DBSCAN). [Largely untested / unused]
                
                The program does the following;
                1. Reads data in from a file. The data must be in csv format.
                2. Applies agglomerative clustering with a user-defined linkage
                3. Applies DP-GMM with user-defined gamma and component numbers
                4. Applies DBSCAN (all default properties)
                5. Create a dendrogram of the agglomerative clusters [OPTIONAL]
                6. Create a cluster overlay on a t-SNE plot for any of the 
                    methods [OPTIONAL]
                7. Apply DP-GMMs to a t-SNE plot and render as a scatterplot
                    overlaid with GMM ellipses [OPTIONAL]
                8. Save any / all of the cluster outputs to csv file [OPTIONAL]
               10. Calculate silhouette score for each method.
               11. Print results and model parameters to the console.
                
Notes:          a) The program is functional, but not well-tested or robust. 
                b) Input and output file paths are HARD-CODED(!!)
                c) DBSCAN 'works', but has undergone very little testing.
****************************************************************************"""

## ****************************************************************************
## * IMPORT RELEVANT MODULES
## ****************************************************************************
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import itertools
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from scipy import linalg
from sklearn.cluster import DBSCAN
from sklearn import mixture
from sklearn.metrics import silhouette_score
from sys import exit

## ****************************************************************************
## * CLUSTERING CLASS - MAIN BODY
## ****************************************************************************
class Clustering():    
    def __init__(self, f_in, f_dendro, f_hclust, f_dpgmm, f_dbscan, 
                 f_tsne=None):
        """
        Constructor. File paths are encapsulated, and data is read from file 
        into a dataframe (also encapsulated).
        """
        self.f_in = f_in
        self.f_dendro = f_dendro
        self.f_hclust = f_hclust
        self.f_dpgmm = f_dpgmm
        self.f_dbscan = f_dbscan
        self.f_tsne = f_tsne
        self.data = self.read_data(f_in)
            
    def read_data(self, f):
        """
        Read data from file (assumes .csv format).
        
        Parameters:
            - f_in, string, the input filename
        
        Return: 
            - pandas dataframe containing the data, with subreddit as index
        """
        try:
            df = pd.read_csv(f)
            return df.set_index('subreddit')
        except FileNotFoundError:
            print("The file " + f + " was not found")
            exit(1)
    
    def hierarchical_clustering(self, method):
        """
        Apply hierarchical clustering using the scipy linkage() function.
        
        Code developed from:
        https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
        
        Parameters:
            - method, string, the linkage method to be used
        
        Return:
            - dendro, the trained hierarchical cluster model.
        """
        # cluster the data using 'method' linkage 
        dendro = linkage(self.data, method)
        # copehenet score is a measure of 'goodness of fit' for the clustering
        c, coph_dists = cophenet(dendro, pdist(self.data))
        print("Cophenet coefficient: "+ str(c))
        # return clustered model
        return dendro
    
    def get_clusters_from_tree(self, dendro, cut_height=None):
        """
        For a clustered model returned by linkage(), apply a cut and to the 
        dendrogram and get all the clusters below that cut. Uses the scipy
        fcluster() function.
        
        Parameters:
            - dendro, a clustered linkage() model
            - cut_height, numeric, the distance at which to place the cut.
        
        Return:
            - dataframe; each subreddit (string) is assigned to a cluster (int)
        """
        if cut_height is not None:
            # get array of cluster allocations
            clusts = fcluster(dendro, cut_height, criterion='distance')
            # create datafrme with cluster allocations and subreddit names
            df = pd.DataFrame({'subreddit': self.data.index.values, 
                               "cluster":clusts}).set_index('subreddit')
            
            print("Estimated number of AHC clusters: " + str(len(set(clusts))))        
            return df
        else:
            print("WARNING: no cut-height supplied to get_clusters_from_tree")
    
    def dbscan(self, ):
        """
        Cluster the data using the DBSCAN method. Uses the DBSCAN() function 
        from scikit-learn, with all parameters as defaults.
                        
        Return:
            - dataframe; each subreddit (string) is assigned to a cluster (int)
        """
        # fit the dbscan model to the data
        db_s = DBSCAN().fit(self.data)
        # get cluster labels
        labels = db_s.labels_
        
        # Number of clusters in labels, ignoring noise if present.
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # assign labels to subreddits
        df = pd.DataFrame({'subreddit':self.data.index.values,
                            'cluster':labels}).set_index('subreddit')
        
        print('Estimated number of DBSCAN clusters: %d' % num_clusters)
        return df

    def dirichlet_gmm(self, seed=1, gmm_cmpts=10, prior=1e-3, 
                      plot_clusters=False):
        """
        Cluster the data using the Dirichlet Process Gaussian Mixtures method. 
        Approximates an infinite mixture model with a finite one, using the 
        stick-breaknig process. Implemented using the scikit-learn 
        BayesianGaussianMixture() function.
        
        Method developed from:
        http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_sin.html#sphx-glr-auto-examples-mixture-plot-gmm-sin-py
        
        Parameters:
            - seed, int, the random number seed (for repeatability)
            - gmm_cmpts, int, the max number of Gaussian distributions to use
            - prior, float, the Dirichlet concentration of each component. 
                Usually referred to as gamma.
            - plot_clusters, boolean, flag to indicate whether to plot the 
                output. USE WITH CAUTION!! Will only work if data is 2D.
            
        Return:
            - dataframe; each subreddit (string) is assigned to a cluster (int)
        """
        # train the DP-GMM
        dp_gmm = mixture.BayesianGaussianMixture(
                n_components=gmm_cmpts, 
                covariance_type='full', 
                n_init=10,      # run the model 10 times, and take the best run
                weight_concentration_prior=prior,
                weight_concentration_prior_type='dirichlet_process',
                mean_precision_prior=prior, 
                init_params="random", 
                random_state=seed).fit(self.data)
        
        # generate cluster labels
        clusts=dp_gmm.predict(self.data)    
        print("Estimated number of DP-GMM clusters: " + str(len(set(clusts))))
        
        # If required, plot the clusters on a 2D scatterplot with ellipses to
        # show Gaussian components.
        #************************************************
        # *** CAUTION!! WILL ONLY WORK WITH 2D DATA!! ***
        #************************************************
        if plot_clusters:
            X = np.array(self.data.reset_index()[['x','y']])
            self.plot_dpgmms(X, clusts, dp_gmm.means_, dp_gmm.covariances_, 1,
                 "Bayesian GMM with a Dirichlet process prior")
        
        # assign labels to subreddits and return
        df = pd.DataFrame({'subreddit': self.data.index.values, 
                           'cluster':clusts}).set_index('subreddit')
        return df        

    def plot_dpgmms(self, X, Y, means, covariances):
        """
        Plot DP-GMMs on a 2D scatterplot, drawing the components as ellipses.
        *** CAUTION: ONLY use 2D data, NO ERRORS ARE HANDLED for other data ***
        
        Method developed from:
        http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_sin.html#sphx-glr-auto-examples-mixture-plot-gmm-sin-py
        
        Parameters:
            - X, dataframe or array, feature set
            - Y, array / list / Series of ints, cluster labels
            - means, array / df of component means
            - covariances, array of covariance matrices
        """
        # create colour palette for ellipses
        color_iter = itertools.cycle(['navy', 'aqua', 'magenta', 'gold',
                                      'darkorange', 'beige', 'coral', 'grey',
                                      'lavender', 'ivory'])
        # create figure, add subplot
        fig = plt.figure()        
        ax = fig.add_subplot(111, aspect='equal')
        # iterate through each mean, covariance and colour 
        for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
            # calculate eigenvectors and normal vectors (ellipse orienation)
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y == i):
                continue
            plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)
    
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            # make the ellipse semi-transparent, to see the points beneath
            ell.set_alpha(0.5)
            ax.add_artist(ell)
            
    def plot_dendrogram(self, dendro, title, orient):
        """
        Plot a dendrogram of the agglomerative hierarchical cluster tree.
        
        Parameters:
            - dendro, linkage modelm used to plot the tree
            - title, string, graph title
            - orient, string, orientation of the dendrogram
        """
        
        # set parameters based on orientation; width and height of plot,
        # labels for axes, and rotation of leaf nodes.
        if orient == 'left' or orient == 'right':
            w = 10; h = 15
            x_lab = 'distance'; y_lab = 'subreddits'
            leaf_rot = 0.0
        elif orient == 'top' or orient == 'bottom':
            w = 15; h = 10
            x_lab = 'subreddits'; y_lab = 'distance'            
            leaf_rot = 90.0            
        else:
            print("Error: 'dendrogram' function does not allow orientation " + orient)
        
        # plot the dendrogram. The plot figure size needs to be large enough 
        # that opening it in a browser doesn't compress the label names s.t. 
        # they're unreadable (generally managed through saving as .svg)
        plt.figure(figsize=(w, h))
        plt.title(title); plt.xlabel(x_lab); plt.ylabel(y_lab)
        dendrogram(
            dendro,
            orientation=orient,
            leaf_rotation=leaf_rot,  
            leaf_font_size=2.,  # font size for the x axis labels
            labels=self.data.index.values
        )
        # save the plot
        plt.savefig(self.f_dendro)
    
    def plot_cluster_overlays(self, clusts, new_col, txt_font=4, point_size=4):
        """
        Plot subreddit points and names on a 2D scatterplot, and colour them 
        based on the clusters they've been assigned to. The 2D points are taken
        from a t-SNE coordinate transformation.
        
        Parameters:
            - clusts, dataframe, subreddits & the clusters they're assigned to
            - new_col, string, name for 'clusters' column (used in plot title)
            - txt_font, numeric, font size for data point labels 
            - point_size, numeric, size of each data point.
        """
        # read t_sne data (subreddit, x, y)
        if self.f_tsne is not None: 
            tsne = self.read_data(self.f_tsne)
        
        # merge clusters with t_sne, on subreddit column
        df = tsne.reset_index().merge(clusts.reset_index(), on='subreddit')
        # rename the clusters column
        df.rename(columns={'cluster':new_col}, inplace=True)
        
        # get the number of clusters from the dataset
        num_clusts = len(set(df[new_col]))
        
        # plot the points
        plt.figure(figsize=(13, 10))
        ax = plt.axes()
        
        # choose a colourmap based on the number of clusters
        c_map = plt.get_cmap('Paired')
        if num_clusts > 12:
            c_map = plt.get_cmap('tab20')
            
        ax.scatter(x=df['x'],y=df['y'],s=point_size, color=c_map(df[new_col]))
        
        # label each point with the subreddit name
        for i, txt in enumerate(df['subreddit']):
            ax.text(df.iloc[i]['x'], df.iloc[i]['y'], txt, fontsize=txt_font, 
                    color=c_map(df.iloc[i][new_col]))
        
        # add plot title
        plt.title(new_col + " clustering overlaid onto t-SNE coordinates, " + 
                  "producing " + str(len(set(df[new_col]))) + " clusters")
        
        plt.show()
        
    def run_clusters(self, gmm_cmpts=10, prior=1e-3,         
                     cut_height=None, linkage='ward',
                     save_hier=True, save_dpgmm=True, save_dbscan=True,
                     plot_dendrogram=False, plot_dpgmm_ellipses=False, 
                     plot_hclust_overlay=False, plot_dpgmm_overlay=False):
        """
        Main driver for the class. Runs all the methods, saves cluster 
        assignments to file (optional), draws plots (optional) and prints info 
        to the console.
        Parameters:
            - gmm_cmpts, int, max number of GMM components for DP-GMM
            - prior, float, Dirichlet process prior
            - cut_height, int, height of the cut for dendrogram
            - linkage, string, linkage type for agglomerative clustering
            - save_hier, boolean, indicates whether to save agglomerative 
                clustering output to file
            - save_dpgmm, boolean, indicates whether to save DP-GMM output to 
                file
            - save_dbscan, boolean, indicates whether to save dbscan output to 
                file
            - plot_dendrograms, boolean, indicates whether to plot a dendrogram
                of agglomerative clustering output
            - plot_dpgmm_ellipses, boolean, indicates whether to plot ellipses
                showing final DP-GMM components
            - plot_hclust_overlay, boolean, indicates whether to plot an 
                overlay of hierarchical clusters on t-SNE points
            - plot_dpgmm_overlay, boolean, indicates whether to plot an 
                overlay of DP-GMM clusters on t-SNE points                
        """
        
        den = self.hierarchical_clustering(method=linkage)
           
        h_clusts = self.get_clusters_from_tree(den, cut_height)        
        if save_hier: h_clusts.to_csv(self.f_hclust)
        
        dpgmm_clusts = self.dirichlet_gmm(gmm_cmpts=gmm_cmpts, prior=prior)
        if save_dpgmm: dpgmm_clusts.to_csv(self.f_dpgmm)
                
        dbscan_clusts = self.dbscan()
        if save_dbscan: dbscan_clusts.to_csv(self.f_dbscan)
        
        if plot_dendrogram:
            self.plot_dendrogram(den, 'Hierarchical clustering of subreddits', 'top')         

        if plot_hclust_overlay:
            self.plot_cluster_overlays(h_clusts, "Agglomerative")
        
        if plot_dpgmm_overlay:
            self.plot_cluster_overlays(dpgmm_clusts, "DP-GMM")
        
        print("\nSilhouette scores\n-----------------")
        print("AHC:    " + str(silhouette_score(self.data, h_clusts['cluster'])))
        print("DP-GMM: " + str(silhouette_score(self.data, dpgmm_clusts['cluster'])))
        #print("DBSCAN: " + str(silhouette_score(self.data, dbscan_clusts)))

## ****************************************************************************
## * MAIN DRIVER
## ****************************************************************************
def main():
    """
    Main method. Sets up model parameters, input and output files, prints info 
    to console and then runs models.    
    """
    # cluster parameters
    gmm_cmpts = 30
    gmm_prior = 0.01
    cut_ht = 20
    link = 'ward'
    d_type = "1511250908_MergedData_logZeroRescale_PCA_Whitened"
    perplexity = 15
    
    # input files
    path = "../Data/Output/Experiment2/"
    infile = path + d_type + ".txt"
    tsne_file = path + "TSNE/" + d_type+"_p"+str(perplexity)+"_tSNE.csv"
    
    # output files
    dendrogram_file = path + "Images/"+d_type+"_dendrogram.svg"     
    h_clusters_file = path+"AGGLOM/h_clusters_"+d_type+"_d"+str(cut_ht)+"_"+link+".csv"
    gmm_clusters_file = path+"DP_GMM/dpgmm_clusters_"+d_type+"_c"+str(gmm_cmpts)+".csv"
    dbscan_clusters_file = path+"DBSCAN/dbscan_"+d_type+"_clusters.csv"
    
    # print info to console
    print("\nModel run info\n--------------")
    print("GMM components: " + str(gmm_cmpts))
    print("GMM prior     : " + str(gmm_prior))
    print("AHC cut-height: " + str(cut_ht))
    print("AHC linkage   : " + link)
    print("Input data    : " + d_type)
    print("--------------\n")
    
    # create Clustering object
    clust = Clustering(f_in=infile, 
                       f_dendro=dendrogram_file, 
                       f_hclust=h_clusters_file, 
                       f_dpgmm=gmm_clusters_file, 
                       f_dbscan=dbscan_clusters_file, 
                       f_tsne=tsne_file)    
    
    # run clustering algorithms, plot graphs etc.
    clust.run_clusters(gmm_cmpts=gmm_cmpts, 
                       prior=gmm_prior,
                       cut_height=cut_ht,
                       linkage=link, 
                       save_hier=False, 
                       save_dpgmm=True,
                       save_dbscan=False,
                       plot_dendrogram=False,
                       plot_hclust_overlay=False, 
                       plot_dpgmm_overlay=False)
    
if __name__ == '__main__':
    main()