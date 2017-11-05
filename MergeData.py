# -*- coding: utf-8 -*-
"""****************************************************************************
Created on Wed Aug 16 20:49:20 2017

@author:        A. VINCENT
Description:    Program to merge output from several MapReduce programs. 
                Inputs can come as .tsv or .csv files (mainly .tsv), and will 
                (*must!*) be of the form 
                "subreddit", "attribute1", "attribute2",..."attributeN"
                
                Input files *must* also have a header line.
                
Notes:          a) The program is functional, but not well-tested or robust. 
                b) Input and output file paths are HARD-CODED(!!)
****************************************************************************"""

## ****************************************************************************
## * IMPORT RELEVANT MODULES
## ****************************************************************************
import pandas as pd
from numpy import log
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from os import scandir
from time import time
from sys import exit

## ****************************************************************************
## * DATA MERGING CLASS
## ****************************************************************************
class MergeData():
    def __init__(self, inpath, f_out):
        """
        Constructor. Set up file references.
        
        Parameters:
            - inpath, string, the directory to read input files from
            - f_out, string, file to write to 
        """
        self.inpath = inpath
        self.f_out = f_out
        self.files = self.get_filenames(inpath)

    def get_filenames(self, path):
        """ 
        Return a list of the filenames of all files in 'path' directory. Ignore
        directories and symbolic links.
        
        NOTE: assumes that all files in 'path' are useful input.
        """
        
        files = []
        try:
            with scandir(path) as input_dir:            
                for entry in input_dir:
                    if entry.is_file():
                        files.append(entry.name)
        except FileNotFoundError:
            print("The directory " + path + " was not found")
            exit(1)
        
        if files:
            print(str(len(files)) + " files found:")
            for file in files: print("\t"+file)
            return files
        else:
            print("No files found at " + path + "; exiting")
            exit(2)
    
    def get_csv_data(self, infile):
        """
        Read data from file (assumes .csv format).
        
        Parameters:
            - infile, name of the file to read.
        
        Return:
            pandas dataframe containing the data.
        """
        try:
            return pd.read_csv(infile)
        except FileNotFoundError:
            print("The file " +infile + " was not found")
            exit(1)
      
    def merge_on_subreddits(self, df1, df2, attr='subreddit'):
        """
        Merges two dataframes on the attribute.
        
        Parameters:
            - df1, df2, dataframes to merge
            - attr, string, attribute to merge on
        
        Return:
            pandas dataframe containing the merged data.
        """
        if attr in df1.columns and attr in df2.columns:
            return df1.merge(df2, on=attr)
        else: 
            print("Error: cannot merge dataframes on attribute " + attr)
            exit(1)

    def set_threshold(self, df, thresholds):
        """
        Filter the dataframe based on thresholds for particular attributes.
        
        Parameters:
            - df, dataframe to filter data from
            - thresholds, dict of form {"attribute" : filter_value} 
                    
        Return:
            pandas dataframe containing the filtered data.
        """        
        for param in thresholds:
            if param not in df.columns:
                print(param + " is not a column in the dataset. Columns: \n" + 
                      str(df.columns))
            else:
                df = df[df[param] >= thresholds[param]]
        
        return df
    
    def rescale_data(self, df):
        """
        Rescale data for specific attributes. Use either a log scale, or 
        centre the data on 0 and add a small value before 
        applying a log scale.
        
        Parameters:
            - df, dataframe to rescale
        
        Return:
            pandas dataframe containing the rescaled data.
        """
        
        log_cols = ['comment_count', 
                    'post_count', 
                    'contributor_count'] 
        log_offset_cols = ['average_num_urls', 
                           'comments_per_post_mean',
                           'average_comment_character_count', 
                           'average_post_character_count',
                           'average_subscriber_count',
                           'media_embed_count'] 
        
        for col in log_cols:
            df[col] = log(df[col])
        
        OFFSET = 0.001
        for col in log_offset_cols:
            df[col] = log(df[col] + abs(df[col].min()) + OFFSET)
        
        return df
    
    def apply_pca(self, df, num_pcs, use_whiten=False):
        """
        Apply Principal Components Analysis to dataframe.
        
        Parameters:
            - df, dataframe to apply PCA to
            - num_pcs, int, the number of principal componetns to use
            - use_whiten, boolean, flag to indicate whether data should be 
                whitened
        
        Return:
            pandas dataframe containing the transformed data.
        """
        # ensure data is scaled appropriately
        subreddits = df.index
        df = StandardScaler().fit_transform(df)
        # create the PCA object with the desired number of principal components
        pca = PCA(n_components = num_pcs, whiten=use_whiten)
        # return the PCA'd data
        df = pd.DataFrame(pca.fit_transform(df))
         
        df.columns = ["PC_" + str(col) for col in df.columns.values]
        return df.set_index(subreddits)
    
    def plot_distributions(self, df):
        """
        Plot distributions of attributes as histogram
        
        Parameters:
            - df, dataframe containing attributes to plot
        """
        for col in df.columns.values:
            plt.figure(); plt.hist(df[col], bins=200)
            plt.title("Histogram showing distribution of " + str(col))
    
    def print_params(self, thresholds, rescale, use_pca, whiten, save_file, 
                     f_out):
        """
        Print a series of parameters to console.
        
        Parameters:
            - thresholds, dict of form {"attribute" : filter_value}
            - rescale, boolean, indicator on whether data has been rescaled
            - use_pca, boolean, indicator on whether data has been PCA'd
            - whiten, boolean, indicator on whether data has been whitened
            - save_file, boolean, indicator ref saving data to file
            - f_out, string, output file                    
        """
        print("\nParameters Used\n---------------")
        if thresholds is not None:
            for t in thresholds:
                print('{:20s} {:7d}'.format(t+" min:", thresholds[t]))
        print('{:20s} {:>7}'.format("Rescale values:", str(rescale)))
        print('{:20s} {:>7}'.format("Use PCA:", str(use_pca)))
        print('{:20s} {:>7}'.format("Use PCA whitening:", str(whiten)))
        if save_file:
            print('{:20s} {}'.format("Output file:", f_out))
        else:
            print('{:20s} {:>7}'.format("Output file:", str(save_file)))
            
    def merge_data(self, thresholds=None, rescale=False, use_pca=False, 
                   pca_whiten=False, plot_dists=False, show_params=True, 
                   write_file=True):
        """
        Main driver for class. Does the following:
        1. Reads data from several .csv files produced by MapReduce and then
            cleaned using CleanData.py
        2. Merges all files together on subreddit name (using inner join).
        3. Filter using thresholds; rescale; apply PCA; apply whitening; plot
            data distributions, print parameters to console, write data to file
            [ALL OPTIONAL]
        
        Parameters:
            - thresholds, dict of form {"attribute" : filter_value}
            - rescale, boolean, indicator on whether data has been rescaled
            - use_pca, boolean, indicator on whether data has been PCA'd
            - pca_whiten, boolean, indicator on whether data has been whitened
            - plot_dists, boolean, indicator on whether data should be plotted
            - show_params, boolean, indicator on whether data should be printed
            - write_file, boolean, indicator ref saving data to file
        """
        # load the data from the files
        for i in range(len(self.files)):
            # first file in the list
            f_name = self.inpath + self.files[i]
            if i == 0:
                df = self.get_csv_data(f_name)            
            else: 
                df = self.merge_on_subreddits(df, self.get_csv_data(f_name))
            
        # apply thresholds (e.g. the min number of comments, posts for a
        # subreddit to be retained)
        if thresholds is not None: df = self.set_threshold(df, thresholds)            
        
        # apply log rescaling to the dataframe
        if rescale: df = self.rescale_data(df)
        
        # set the index to be the 'subreddit' column
        df.set_index("subreddit", inplace=True)        
        
        # perform PCA on the data
        if use_pca: df = self.apply_pca(df, len(df.columns.values), 
                                        use_whiten=pca_whiten)
        
        # plot histograms showing distributions of each attribute (if required)
        if plot_dists: self.plot_distributions(df)  
        
        # print the parameters used to the console
        if show_params: self.print_params(thresholds, rescale, use_pca, 
                                          pca_whiten, write_file, self.f_out)
        
        # write to csv, or print df head to console
        if write_file: 
            df.to_csv(self.f_out)        
        else:
            print("\nData snapshot\n-------------")
            print(df.head())

## ****************************************************************************
## * HELPER FUNCTIONS
## ****************************************************************************
def generate_output_filename(use_pca=True, whiten=True, rescale=True):
    
    base = str(int(time())) + "_MergedData"

    if use_pca:
        base += "_PCA"
        if whiten:
            base += "_Whitened"
        else:
            base += "_notWhitened"
    else:
        base += "_noPCA"
    
    if rescale:
        base += "_Rescaled"
    else:
        base += "_notRescaled"
        
    base += ".txt"
    
    return base
    
## ****************************************************************************
## * MAIN METHOD
## ****************************************************************************
def main():
    """
    Main driver for program. Set filenames and parameters, then call MergeData.
    """
    
    input_path = "../Data/Input/Processed/"
    
    rescale_data=True
    use_pca=False
    whiten=False
    
    outfile = "../Data/Output/"+generate_output_filename(use_pca=use_pca,
                                                         whiten=whiten,
                                                         rescale=rescale_data)
    
    parameter_thresholds = {'comment_count':100000, 'post_count':1000}
    
    md = MergeData(input_path, outfile)
    
    md.merge_data(thresholds=parameter_thresholds, 
                  rescale=rescale_data,
                  use_pca=use_pca,
                  pca_whiten=whiten,
                  plot_dists=True,
                  write_file=True)
    
if __name__ == '__main__':
    main()