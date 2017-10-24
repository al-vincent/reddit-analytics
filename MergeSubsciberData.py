# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 17:21:38 2017

@author: Al
"""

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
from os import scandir
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
            df = pd.read_csv(infile).set_index('subreddit')
            return df.dropna()
        except FileNotFoundError:
            print("The file " +infile + " was not found")
            exit(1)
    
    def merge_data(self, write_file=True):
        """
        Main driver for class. Does the following:
        1. Reads data from several .csv files produced by MapReduce and then
            cleaned using CleanData.py
        2. Merges all files together on subreddit name (using inner join).
        
        Parameters:            
            - write_file, boolean, indicator ref saving data to file
        """
        # load the data from the files
        for i in range(len(self.files)):
            # first file in the list
            f_name = self.inpath + self.files[i]
            if i == 0:
                df = self.get_csv_data(f_name)            
            else: 
                df = pd.concat([df, self.get_csv_data(f_name)])      
        
        df.dropna(inplace=True)
        # write to csv, or print df head to console
        if write_file: 
            df.to_csv(self.f_out)        
        else:
            print("\nData snapshot\n-------------")
            print(df.head())
    
## ****************************************************************************
## * MAIN METHOD
## ****************************************************************************
def main():
    """
    Main driver for program. Set filenames and parameters, then call MergeData.
    """
    
    path = "../Data/Input/Processed/SubscriberCounts/"     
    outfile = path+"Output/merged.txt"
           
    md = MergeData(path, outfile)    
    md.merge_data(write_file=True)
    
if __name__ == '__main__':
    main()