# -*- coding: utf-8 -*-
"""****************************************************************************
Created on Wed Aug 16 12:01:33 2017

@author:        A. VINCENT
Description:    Program to plot distributions of subreddit features.
                
                Outputs a series of histograms for each feature in dataset.
                
Notes:          a) The program is functional, but not well-tested or robust. 
                b) Input and output file paths are HARD-CODED(!!)
****************************************************************************"""

## ****************************************************************************
## * IMPORT RELEVANT MODULES
## ****************************************************************************
import pandas as pd
from numpy import log
import matplotlib.pyplot as plt
from sys import exit

## ****************************************************************************
## * PLOTTING CLASS
## ****************************************************************************
class PlotDistributions():
    def __init__(self, infile):
        """
        Constructor. Encapsulates input file, and reads and encapsulates data.
        """
        self.infile = infile
        self.data = self.read_data()
    
    def read_data(self):
        """
        Read data from csv input file.
        
        Return a pandas dataframe containing the data.
        """
        try:
            df = pd.read_csv(self.infile)
            # set the index to be the names of the subreddits
            return df.set_index('subreddit')    
        except FileNotFoundError:
            print("The file " + self.infile + " was not found")
            exit(1)
            
    def plot_all_hists(self, n_bins=50):
        """
        Plots a histogram of every feature in the dataset.
        
        Parameters:
            - n_bins, int, the number of bins to use
        """
        for col in self.data.columns:
            t = "Distribution of values in " + col + ", per subreddit"
            plt.figure(); self.data[col].plot.hist(bins=n_bins, title=t)
    
    def plot_histogram(self, col_name, n_bins=50):
        """
        Plots a histogram of a single feature in the dataset.
        
        Parameters:
            - col_name, string, the name of the feature to plot.
            - n_bins, int, the number of bins to use
        """
        # plot a histogram of the distribution of 'col_name'
        if col_name in self.data.columns:
            t = "Distribution of " + col_name
            plt.figure(); self.data[col_name].plot.hist(bins=n_bins, title=t)
        
    
    def plot_log_histogram(self, col_name, n_bins=50):
        """
        Plots a histogram of the log of a single feature in the dataset. If any
        values in the feature are negative, the whole feature is shifted by 
        abs(feature_min) + 0.01
        
        Parameters:
            - col_name, string, the name of the feature to plot.
            - n_bins, int, the number of bins to use
        """
        if col_name in self.data.columns:            
            t = "Distribution of log(" + col_name + ")"
            if self.data[col_name].min() > 0:                
                ser = self.data[col_name].apply(log)
                plt.figure(); ser.plot.hist(bins=n_bins, title=t)
            else:
                print("** WARNING: " + col_name + " includes zero values. **" +
                      "\n** Data will be plotted with a small value added. **")
                add_on = abs(self.data[col_name].min()) + 0.001                 
                ser = log(self.data[col_name] + add_on)
                plt.figure(); ser.plot.hist(bins=n_bins, title=t + ", scaled by 0.01")
        else:
            print("The attribute " + col_name + " is not contained in the input file.")
    
    def run_plot_dists(self):
        """
        Driver program for the plotter.
        """
        self.plot_all_hists()
        self.plot_histogram('flesch_kincaid')
        self.plot_log_histogram('contributor_count')
        self.plot_log_histogram('average_num_urls')
        self.plot_histogram('pc_link_posts')
        self.plot_log_histogram('comment_count'),
        self.plot_log_histogram('post_count')
        self.plot_log_histogram('comments_per_post_mean')
        self.plot_log_histogram('average_character_count')

## ****************************************************************************
## * MAIN METHOD
## ****************************************************************************
def main():
    infile = "../Data/Output/MergedData_noPCA_noRescale.txt"
    plot_dists = PlotDistributions(infile)  
    plot_dists.run_plot_dists()

if __name__ == '__main__':
    main()