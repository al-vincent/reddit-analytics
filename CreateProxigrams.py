# -*- coding: utf-8 -*-
"""****************************************************************************
Created on Mon Sep 11 07:48:32 2017

@author:        A. VINCENT
Description:    Program to create proxigrams from feature data and 2D transformed
                coordinates (transformed from *the same* feature data).
                
                Outputs a plot showing every subreddit cast onto 2D transformed
                coords, with arrows to its k nearest neighbouts. Lines are 
                coloured by how close (Euclidean distance) points are.
                
Notes:          a) The program is functional, but not well-tested or robust. 
                b) Input and output file paths are HARD-CODED(!!)
****************************************************************************"""

## ****************************************************************************
## * IMPORT RELEVANT MODULES
## ****************************************************************************
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import SpectralEmbedding, MDS, TSNE, Isomap, LocallyLinearEmbedding
import matplotlib.pyplot as plt
from sys import exit
from os import scandir
from time import time

class Constants():
    def __init__(self):
        self.tsne = "tsne"
        self.mds = "mds"
        self.spectral = "spectral"
        self.isomap = "isomap"
        self.lle = "lle"

class CreateProxigram():
    def __init__(self, f_data, f_coords=None, testing=False, use_coord_file=False, 
                 draw_proxigram=True, plot_lines=True, plot_name=None, k=3):
        self.f_data = f_data
        self.f_coords = f_coords
        self.testing = testing
        self.use_coord_file = use_coord_file
        self.show_plot = draw_proxigram
        self.plot_lines = plot_lines
        self.plot_name = plot_name
        self.k = k
        self.method = None
        self.plot_title = None
        self.df = self.read_data()
        
    def read_data(self):
        # get dataset
        try:
            return pd.read_csv(self.f_data, index_col='subreddit')
        except FileNotFoundError:
            print("The file "+self.f_data+" was not found; exiting.")
            exit(1)  

    #def get_knn(self, df):  
    def get_knn(self):  
        """
        Calculate the k nearest neighbours for each point in df, using Eucildean 
        distance.
        
        Parameters:
            - df, dataframe of subreddits and features.
            - k, int, number of nearest neighbours to capture.
        
        Return:
            nearest_k, n * k array (n is number of data points) of *indices* of k 
                nearest neighbours
            dists, n * n array of subreddit distances
            minimum and maximum distances (floats)
        """
        # get a (square) distance matrix between all subreddits
        dists = squareform(pdist(self.df))
        
        # change all the zeros in the leading diag to NaNs (helpful later!)
        rows, cols = dists.shape
        for r in range(rows):
            for c in range(cols):
                if r == c: dists[r][c] = np.NaN
        
        # partition the distance matrix so that the k smallest distances are moved
        # to cols 1,..,k (or 0,..,(k-1))
        nearest_k = np.apply_along_axis(np.argpartition, 1, dists, self.k)
    
        # drop the remaining columns (unnecessary)
        return nearest_k[:,0:self.k], dists, np.nanmin(dists), np.nanmax(dists)
        
    def convert_knn_to_dict(self, dists, nearest_k, names):
        """
        Convert the k nearest neighbour arrays to a dictionary-of-dictionaries, so 
        that arrows can be indexed and drawn easily.
        
        Parameters:
            - dists, n * n distance matrix
            - nearest_k, n * k array of knn indices for each subreddit
            - names, list of strings, names of subreddits.
        
        Return:
            proxis. A dict-of-dicts, that looks like this;
            {subreddit#1 : 
                {knn1 : <distance>, knn2 : <distance>, knn3 : <distance>}
            subreddit#2 : 
                {knn1 : <distance>, knn2 : <distance>, knn3 : <distance>}
            ...etc.
            }
        """
        rows, cols = nearest_k.shape
        proxis = {}
        for row in range(rows):
            row_proxis = {}
            for col in range(cols):
                if names[row] != names[nearest_k[row][col]]:
                    row_proxis[names[nearest_k[row][col]]] = dists[row][nearest_k[row][col]]
            proxis[names[row]] = row_proxis
        
    #    print("\ndists\n-----"); print(dists)
    #    print("\nnearest_k\n-----"); print(nearest_k)
    #    print("\nnames\n-----"); print(names)
    #    print("\nproxis\n------"); print(proxis)
        
        return proxis
    
    #def plot_data(self, coords, df, proxi_dists, d_min, d_max, names=None):
    def plot_data(self, coords, proxi_dists, d_min, d_max, names=None):
        """
        Plot the proxigram. Create a scattergraph using 2D points and 
        subreddit names, then plot arrows between each point and its knn. 
        Colour arrows based on distance (in feature space, *not* t-SNE space).
        
        Parameters:
            - coords, dataframe of 2D (x,y) coords for each subreddit
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
        
        # create a scatterplot showing subreddits on 2D coordinates
        plt.figure(figsize=(13, 10))
        ax = plt.axes()    
        ax.scatter(x=coords['x'],y=coords['y'],s=2)#,color='black'
        
        # plot lines between each subreddit and its k nearest neighbours
        if self.plot_lines:
            for subred in proxi_dists:
                # get the start (x,y) coords, using the subred key into df
                start = coords.loc[subred].tolist()
                for neighbour in proxi_dists[subred]:
                    # get the end (x,y) coords, using the second dict keys
                    end = coords.loc[neighbour].tolist()            
                    # colour the arrow, based on the value of the distance 
                    c_map = plt.get_cmap('jet')
                    # this rescales the colour so it isn't all shades of red
                    colours = abs((np.log(proxi_dists[subred][neighbour]) - 
                                   np.log(d_max)) / (np.log(d_min) - np.log(d_max)))
                    #  draw an arrow between the two points
                    ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                             fc=c_map(colours), ec=c_map(colours), 
                             length_includes_head=True, linewidth=0.3)           
                    
        # add subreddit names as labels to points
        if names is not None:
            for i, txt in enumerate(names):            
                ax.text(coords.iloc[i][0], coords.iloc[i][1], txt, fontsize=2)
    
        if self.show_plot:
            plt.title(self.plot_title)
            plt.show()
        else:
            if self.plot_name is not None:
                plt.savefig(self.plot_name)
            else:
                plt.savefig(str(int(time())) + ".svg")
            plt.close()
    
#    def get_coords(self, df=None, names=None, perplex=50, spectral_neighbours=1, 
#                   iso_neighbours=5, lle_neighbours=5, metric_mds=True, seed=1):
    def get_coords(self, names=None, perplex=50, spectral_neighbours=1, 
                   iso_neighbours=5, lle_neighbours=5, metric_mds=True, seed=1):
        """
        Load transformed coordinates from file, or generate using scikit-learn (takes 
        longer).
        
        Parameters:
            - df, dataframe of original subreddit feature values
            - names, list of subreddit names (strings)
            - perplex, int, perplexity level to use
            - seed, int, the random seed
            
        Return:
            dataframe of subreddits and their transformed (x,y) coordinates, with 
            subreddit names as the index.
        """
        
        coords = None
        # if there's a 2D coords filename, try to load the data. Print error if 
        # file not found.
        if self.f_coords is not None:
            try:
                coords = pd.read_csv(self.f_coords) 
                self.plot_title = "Proxigram using t-SNE, perplexity " + str(perplex)
            except FileNotFoundError:
                print("The coordinate file " + self.f_coords + " was not found")
                exit(1)
        # otherwise, run the scikit-learn algorithm 
        else:
            if self.df is not None and names is not None:
#                coords = TSNE(perplexity=perplex, n_iter=5000, 
#                            random_state=seed).fit_transform(df);             
#                coords = pd.DataFrame({"subreddit":names, 
#                                     "x":coords[:,0], "y":coords[:,1]})
                if self.method == "tsne":
                    coords = self.run_tsne(data=self.df, perplexity=perplex, seed=seed)
                    self.plot_title = "Proxigram using t-SNE, perplexity " + str(perplex)
                elif self.method == "spectral":
                    coords = self.run_spectral_embedding(data=self.df, seed=seed, 
                                                         neighbours=spectral_neighbours)
                    self.plot_title = ("Proxigram using spectral embedding, neighbours " 
                                       + str(spectral_neighbours))
                elif self.method == "mds":
                    coords = self.run_mds(data=self.df, seed=seed, metric=metric_mds)
                    self.plot_title = ("Proxigram using mds. Using metric MDS: " 
                                       + str(metric_mds))
                elif self.method == "isomap":
                    coords = self.run_isomap(data=self.df, n_neighbours=iso_neighbours)
                    self.plot_title = ("Proxigram using isomap, with " 
                                       + str(iso_neighbours) + " neighbours")
                elif self.method == "lle":
                    coords = self.run_lle(data=self.df, n_neighbours=iso_neighbours, seed=seed)
                    self.plot_title = ("Proxigram using locally linear embedding, with " 
                                       + str(lle_neighbours) + " neighbours")
                    
            else:
                print("'df' and / or 'names' is missing; both are required.")
                exit(2)
        return coords.set_index('subreddit')
    
    def run_spectral_embedding(self, data, neighbours=1, seed=1):
    
        coords= SpectralEmbedding(random_state=seed, n_neighbors=neighbours,
                                  eigen_solver="arpack", n_jobs=-1).fit_transform(data)
        
        return pd.DataFrame({"subreddit":data.index.values, 
                             "x":coords[:,0], "y":coords[:,1]})

    def run_mds(self, data, metric=True, seed=1):
            
        coords = MDS(n_jobs=-1, max_iter=1000, metric=metric, 
                     random_state=seed).fit_transform(data)
        
        return pd.DataFrame({"subreddit":data.index.values, 
                             "x":coords[:,0], "y":coords[:,1]})
    
    def run_tsne(self, data, perplexity=50, seed=1):
        coords = TSNE(random_state=seed, n_iter=5000, 
                      perplexity=perplexity).fit_transform(data)
        
        return pd.DataFrame({"subreddit":data.index.values, 
                             "x":coords[:,0], "y":coords[:,1]})
                
    def run_isomap(self, data, n_neighbours=5):
        coords = Isomap(n_neighbors=n_neighbours, n_jobs=-1).fit_transform(data)
        
        return pd.DataFrame({"subreddit":data.index.values, 
                             "x":coords[:,0], "y":coords[:,1]})
    
    def run_lle(self, data, n_neighbours=5, seed=1):
        coords = LocallyLinearEmbedding(n_neighbors=n_neighbours, random_state=seed,
                                        n_jobs=-1).fit_transform(data)
        
        return pd.DataFrame({"subreddit":data.index.values, 
                             "x":coords[:,0], "y":coords[:,1]})
    
    #def cut_datasets(self, n, df, coords, names):
    def cut_datasets(self, n, coords, names):
        """
        Take a small cut of n lines of the subreddit feature data, the 2D
        coordinate data and the list of subreddit names. 
        [Useful for generating small proxigrams and checking their output]
        
        Parameters:
            - n, the number of lines to get
            - df, dataframe of original subreddit feature values
            - coords, dataframe of subreddits and their 2D embedded coordinates
            - names, list of subreddit names (strings) 
        
        Return:
            the first n lines of each parameter
        """
        self.df = self.df.head(n)
        coords = coords.head(n)
        names = names[0:n]
        return coords, names
    
    def create_proxigram(self, method="tsne", perplexity=50, spectral_neighbours=1, 
                         iso_neighbours=5, lle_neighbours=5, metric_mds=True, seed=1):
        """
        Driver function to create a single proxigram from a datafile. Can either 
        use (x,y) data generated externally and read in from a file, or use the 
        relevant sklearn function.
        """
        
        assert method in ["tsne", "mds", "spectral", "isomap", "lle"]
        self.method = method
        
#        # get dataset
#        try:
#            df = pd.read_csv(self.f_data, index_col='subreddit')
#        except FileNotFoundError:
#            print("The file "+self.f_data+" was not found; exiting.")
#            exit(3)    
        
        # get a list of subreddit names        
        names = self.df.index.values.tolist()    
        
        # get coordinate dataframe, either from file or generated by sklearn 
        if self.use_coord_file: # to run from file
            coords = self.get_coords(perplex=perplexity) 
        else: # to generate coords using sklearn 
            print("NOTE: Using scikit-learn to generate 2D coordinates")
            coords = self.get_coords(names=names, perplex=perplexity, 
                                     spectral_neighbours=spectral_neighbours, 
                                     iso_neighbours=iso_neighbours, 
                                     lle_neighbours=lle_neighbours,
                                     metric_mds=metric_mds, seed=seed) 
        
        # If testing=true, the program can be run with a small dataset of n lines
        #if self.testing: df, coords, names = self.cut_datasets(20, df, coords, names)
        if self.testing: coords, names = self.cut_datasets(20, coords, names)
                
        # get subreddit knn, distances and the largest, smallest distances 
        knn, dists, d_min, d_max = self.get_knn()
        
        # create an efficient, iterable structure for finding each subreddit's knn
        proxi_dists = self.convert_knn_to_dict(dists, knn, names)  
        
        # plot the proxigrams
        #self.plot_data(coords, df, proxi_dists, d_min, d_max, names=names)
        self.plot_data(coords, proxi_dists, d_min, d_max, names=names)
    
## ****************************************************************************
## * HELPER FUNCTIONS
## ****************************************************************************
def get_input_files(path):
    """ 
    Return a list of the filenames of all files in 'path' directory. Ignore
    directories and symbolic links.
    
    NOTE: assumes that all files in 'path' are useful input.
    """
    
    # find all files in 'path' and store the filnames in the 'files' list
    files = []
    try:
        with scandir(path) as input_dir:            
            for entry in input_dir:
                if entry.is_file():
                    files.append(entry.name)
    except FileNotFoundError:
        print("The directory " + path + " was not found")
        exit(4)
    
    # print names of files found to console
    if files:
        print(str(len(files)) + " files found:")
        for file in files: print("\t"+file)
        return files
    else:
        print("No files found at " + path + "; exiting")
        exit(5)

def create_many_proxigrams(path, testing=False, use_coord_file=False, 
                           draw_proxigram=False, plot_lines=True, method="tsne",
                           perplexity=50, k=3):
    """
    Create a proxigram for each input file found in the 'path' directory. Default 
    behaviour is to save each proxigram as a SVG file, and *not* show the 
    proxigram on the screen. 
    
    NOTES: 
        - If coordinate data is to be read from a file, then filename *MUST* be the 
        same as the data input but with '_tSNE.csv' appended.
        E.g. if the data file is named a.txt, the t-SNE file must be a_tSNE.csv
        - The name of the SVG file will be the same as the name of the t-SNE file,
        but with an .svg file extension (rather than .csv). It will be saved in 
        an 'Images' subfolder in 'path'.
    """
    # get list of files in 'path'
    files = get_input_files(path)
    
    print("\nFiles written:")
    # iterate through each data input file
    for f in files:
        # create relevant filenames from data input file
        if use_coord_file: 
            f_coords = f[:len(f)-4]+"_p"+str(perplexity)+"_wt_tSNE.csv"
        plot_name = f_coords[:len(f_coords)-4] + ".svg"
        # create proxigram from data and t-SNE files
        proxi = CreateProxigram(f_data=path+f,
                                f_coords=path+"TSNE/"+f_coords,
                                testing=testing, 
                                use_coord_file=use_coord_file,
                                draw_proxigram=draw_proxigram, 
                                plot_lines=plot_lines, 
                                plot_name=path + "Images/" + plot_name,
                                k=k)
        if method == 'tsne':
            proxi.create_proxigram(method=method, perplexity=perplexity)
        ## AV: these are all placeholders; none of the manifold learning 
        ## methods seem to perform as well as t-SNE, so I'm leaving them 
        ## out of the multiple-runs option for now
        elif method == "mds":
            pass
        elif method == "spectral": 
            pass
        elif method == "isomap":
            pass
        elif method == "lle":
            pass
        
        print("\t" + plot_name)

def main(single_run=False):
    """
    Main driver program. Read data from files and create proxigrams or 
    scatterplots, as required.
    """    
    
    # set run parameters
    testing = False
    use_coord_file=True
    draw_proxigram=False
    plot_lines=True
    plot_name=None
    method="tsne"
    tsne_perplexity = 50
    nearest_neighbours = 3
    
    # run the embedding    
    if single_run:
        # input files
        path = "../Data/Output/Experiment2/"
        f_in = path + "1511250908_MergedData_logZeroRescale_PCA_Whitened.txt"
        if use_coord_file: 
            f_coords = path + "TSNE/MergedData_noPCA_Rescaled_p30_tSNE.csv"
        else:
            f_coords = None
            
        # create single proxigram plot    
        proxi = CreateProxigram(f_data=f_in,
                                f_coords=f_coords,
                                testing=testing, 
                                use_coord_file=use_coord_file,
                                draw_proxigram=draw_proxigram, 
                                plot_lines=plot_lines, 
                                plot_name=plot_name,
                                k=nearest_neighbours)
        
        C = Constants()
        proxi.create_proxigram(method=C.tsne, perplexity=tsne_perplexity)
        proxi.create_proxigram(method=C.mds, metric_mds=True)
        proxi.create_proxigram(method=C.spectral, spectral_neighbours=None)
        proxi.create_proxigram(method=C.isomap, iso_neighbours=20)
        proxi.create_proxigram(method=C.lle, lle_neighbours=50)
        
    else:
        # set main path for reading data files and writing output
        file_path = "../Data/Output/Experiment2/"
        # create multiple proxigram plots and save (don't display)
        create_many_proxigrams(path=file_path, 
                               testing=testing, 
                               use_coord_file=use_coord_file, 
                               draw_proxigram=draw_proxigram, 
                               plot_lines=plot_lines,
                               method=method,
                               perplexity=tsne_perplexity,
                               k=nearest_neighbours)
    
if __name__ == '__main__':
    main()