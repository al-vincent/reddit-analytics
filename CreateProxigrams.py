# -*- coding: utf-8 -*-
"""****************************************************************************
Created on Mon Sep 11 07:48:32 2017

@author:        A. VINCENT
Description:    Program to create proxigrams from feature data and 2D t-SNE 
                coordinates (transformed from *the same* feature data).
                
                Outputs a plot showing every subreddit cast onto its 2D t-SNE
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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sys import exit
from os import scandir
from time import time

class CreateProxigram():
    def __init__(self, f_data, f_tsne=None, testing=False, use_tsne_file=False, 
                 draw_proxigram=True, plot_lines=True, plot_name=None, 
                 tsne_perplexity=15, k=3):
        self.f_data = f_data
        self.f_tsne = f_tsne
        self.testing = testing
        self.use_tsne_file = use_tsne_file
        self.show_plot = draw_proxigram
        self.plot_lines = plot_lines
        self.plot_name = plot_name
        self.perplexity = tsne_perplexity
        self.k = k

    def get_knn(self, df):  
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
        dists = squareform(pdist(df))
        
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
    
    def plot_data(self, tsne, df, proxi_dists, d_min, d_max, names=None):
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
        
        # create a scatterplot showing subreddits on t-SNE coordinates
        plt.figure(figsize=(13, 10))
        ax = plt.axes()    
        ax.scatter(x=tsne['x'],y=tsne['y'],s=2)#,color='black'
        
        # plot lines between each subreddit and its k nearest neighbours
        if self.plot_lines:
            for subred in proxi_dists:
                # get the start (x,y) coords, using the subred key into df
                start = tsne.loc[subred].tolist()
                for neighbour in proxi_dists[subred]:
                    # get the end (x,y) coords, using the second dict keys
                    end = tsne.loc[neighbour].tolist()            
                    # colour the arrow, based on the value of the distance 
                    c_map = plt.get_cmap('jet')
                    # this rescales the colour so it isn't all shades of red
                    colours = abs((np.log(proxi_dists[subred][neighbour]) - 
                                   np.log(d_max)) / (np.log(d_min) - np.log(d_max)))
                    #  draw an arrow between the two points
                    ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                             fc=c_map(colours), ec=c_map(colours), 
                             length_includes_head=True, linewidth=0.4)           
                    
        # add subreddit names as labels to points
        if names is not None:
            for i, txt in enumerate(names):            
                ax.text(tsne.iloc[i][0], tsne.iloc[i][1], txt, fontsize=2)
    
        if self.show_plot:
            plt.show()
        else:
            if self.plot_name is not None:
                plt.savefig(self.plot_name)
            else:
                plt.savefig(str(int(time())) + ".svg")
            plt.close()
    
    def get_tsne(self, df=None, names=None, perplex=50, seed=1):
        """
        Load t-SNE coordinates from file, or generate using scikit-learn (takes 
        longer).
        
        Parameters:
            - f_tsne, file containing t-SNE coordinates
            - df, dataframe of original subreddit feature values
            - names, list of subreddit names (strings)
            - perplex, int, perplexity level to use
            - seed, int, the random seed
            
        Return:
            dataframe of subreddits and their t-SNE (x,y) coordinates, with 
            subreddit names as the index.
        """
        
        tsne = None
        # if there's a t-SNE filename, try to load the data. Print error if 
        # file not found.
        if self.f_tsne is not None:
            try:
                tsne = pd.read_csv(self.f_tsne) 
            except FileNotFoundError:
                print("The tSNE file " + self.f_tsne + " was not found")
                exit(1)
        # otherwise, run the scikit-learn t-SNE algorithm 
        else:
            if df is not None and names is not None:
                tsne = TSNE(perplexity=perplex, n_iter=5000, 
                            random_state=seed).fit_transform(df);             
                tsne = pd.DataFrame({"subreddit":names, 
                                     "x":tsne[:,0], "y":tsne[:,1]})
            else:
                print("df and names are both required.")
                exit(2)
        return tsne.set_index('subreddit')
    
    def cut_datasets(self, n, df, tsne, names):
        """
        Take a small cut of n lines of the subreddit feature data, the t-SNE 
        coordinate data and the list of subreddit names. 
        [Useful for generating small proxigrams and checking their output]
        
        Parameters:
            - n, the number of lines to get
            - df, dataframe of original subreddit feature values
            - tsne, dataframe of subreddits and their 2D tsne coordinates
            - names, list of subreddit names (strings) 
        
        Return:
            the first n lines of each parameter
        """
        df = df.head(n)
        tsne = tsne.head(n)
        names = names[0:n]
        return df, tsne, names
    
    def create_proxigram(self):
        """
        Driver function to create a single proxigram from a datafile. Can either 
        use t-SNE data generated externally and read in from a file, or use the 
        sklearn t-SNE function.
        
        Parameters:
            - f_data: string, name of text file containing data to be read. (In 
            this case, will use output from MergeData.py)
            - f_tsne: string, name of text file containing corresponding t-SNE data
            - testing: boolean, flag to indicate whether or not to use the full 
            input dataset or a cut of it. Default: false.
            - use_tsne_file: boolean, flag to indicate whether to take t-SNE coords
            from an external file or generate using sklearn. Default: false.
            - draw_proxigram: boolean, flag to indicate whether to draw the 
            proxigram. If false, proxigram will instead be saved to file
        """
        
        # get dataset
        try:
            df = pd.read_csv(self.f_data, index_col='subreddit')
        except FileNotFoundError:
            print("The file "+self.f_data+" was not found; exiting.")
            exit(3)    
        
        # get a list of subreddit names        
        names = df.index.values.tolist()    
        
        # get tsne dataframe, either from file or generated by sklearn 
        if self.use_tsne_file: # to run from file
            tsne = self.get_tsne(df=None, names=None) 
        else: # to generate sklearn tSNE
            print("NOTE: Using scikit-learn tSNE")
            tsne = self.get_tsne(df=df, names=names, perplex=30) 
        
        # If testing=true, the program can be run with a small dataset of n lines
        if self.testing: df, tsne, names = self.cut_datasets(20, df, tsne, names)
                
        # get subreddit knn, distances and the largest, smallest distances 
        knn, dists, d_min, d_max = self.get_knn(df)
        
        # create an efficient, iterable structure for finding each subreddit's knn
        proxi_dists = self.convert_knn_to_dict(dists, knn, names)  
        
        # plot the proxigrams
        self.plot_data(tsne, df, proxi_dists, d_min, d_max, names=names)
    
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

def create_many_proxigrams(path, testing=False, use_tsne_file=False, 
                           draw_proxigram=False, plot_lines=True, 
                           perplexity=15, k=3):
    """
    Create a proxigram for each input file found in the 'path' directory. Default 
    behaviour is to save each proxigram as a SVG file, and *not* show the 
    proxigram on the screen. 
    
    NOTES: 
        - If t-SNE data is to be read from a file, then filename *MUST* be the 
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
        if use_tsne_file: 
            f_tsne = f[:len(f)-4]+"_p"+str(perplexity)+"_tSNE.csv"
        plot_name = f_tsne[:len(f_tsne)-4] + ".svg"
        # create proxigram from data and t-SNE files
        proxi = CreateProxigram(f_data=path+f,
                                f_tsne=path+"TSNE/"+f_tsne,
                                testing=testing, 
                                use_tsne_file=use_tsne_file,
                                draw_proxigram=draw_proxigram, 
                                plot_lines=plot_lines, 
                                plot_name=path + "Images/" + plot_name,
                                tsne_perplexity=perplexity,
                                k=k)
        proxi.create_proxigram()
        print("\t" + plot_name)

def main(single_run=False):
    """
    Main driver program. Read data from files and create proxigrams or 
    scatterplots, as required.
    """    
    
    # set run parameters
    testing = False
    use_tsne_file=True
    draw_proxigram=False
    plot_lines=True
    plot_name=None
    tsne_perplexity = 15
    nearest_neighbours = 3
    
    # run the embedding    
    if single_run:
        # input files
        f_in = "../Data/Output/Experiment1/1510732479_MergedData_noPCA_notWhitened_Rescaled.txt"
        f_tsne = "../Data/Output/MATLAB/MergedData_noPCA_Rescaled_p30_tSNE.csv"
        
        # create single proxigram plot    
        proxi = CreateProxigram(f_data=f_in,
                                f_tsne=f_tsne,
                                testing=testing, 
                                use_tsne_file=use_tsne_file,
                                draw_proxigram=draw_proxigram, 
                                plot_lines=plot_lines, 
                                plot_name=plot_name,
                                tsne_perplexity=tsne_perplexity,
                                k=nearest_neighbours)
        proxi.create_proxigram()
    else:
        # set main path for reading data files and writing output
        file_path = "../Data/Output/Experiment2/"
        # create multiple proxigram plots and save (don't display)
        create_many_proxigrams(path=file_path, 
                               testing=testing, 
                               use_tsne_file=use_tsne_file, 
                               draw_proxigram=draw_proxigram, 
                               plot_lines=plot_lines, 
                               perplexity=tsne_perplexity,
                               k=nearest_neighbours)
    
if __name__ == '__main__':
    main()