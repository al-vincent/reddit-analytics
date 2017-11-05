# -*- coding: utf-8 -*-
"""****************************************************************************
Created on Fri Oct 13 16:49:35 2017

@author: Al
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
## * DATA CLEANING CLASS
## ****************************************************************************

class FileInfo():
    def __init__(self):
        self.keep_cols = {"AverageCommentCharacterCount.txt":"average_comment_character_count",
                          "AverageNsfwPosts.txt":"average_nsfw",
                          "AveragePostCharacterCount.txt":"average_post_character_count",
                          "AverageURLs.txt":"average_num_urls",
                          "CommentsPerPost.txt":"comments_per_post_mean",
                          "CommentEntropy.txt":"comment_entropy",
                          "ContributorCommentRatio.txt":"contributor_comment_ratio_mean",
                          "CountComments.txt":"comment_count",
                          "CountContributors.txt":"contributor_count",
                          "CountMediaEmbeds.txt":"media_embed_count",
                          "CountPosts.txt":"post_count",
                          "LanguageComplexity.txt":"flesch_kincaid",
                          "PostEntropy.txt":"post_entropy",
                          "PostCommentEntropy.txt":"post_comment_entropy",
                          "PostType.txt":"pc_link_posts"}
        
        self.check_cols = {"PostType.txt":["num_null_posts","pc_null_posts"]}
        
        self.is_csv = ["CommentsPerPost.txt", "ContributorCommentRatio.txt"]

class CleanData():
    def __init__(self, inpath, outpath, files=None):
        """
        Constructor. Set up path references.
        
        Parameters:
            - inpath, path to the raw .txt data files (output from MapReduce)
            - outpath, path to the cleaned .csv data files.
            - files, a list of filenames to clean (default None, in which case
            all files in inpath are cleaned)
        """
        self.inpath = inpath
        self.outpath = outpath
        if files is not None:
            self.files = files
        else:
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
            for f in files: print("\t"+f)
            return files
        else:
            print("No files found at " + path + "; exiting")
            exit(2)
    
    def get_tsv_data(self, infile):
        """
        Read data from file (assumes .tsv-type format, which is consistent
        with MapReduce output).
        
        Parameters:
            - infile, name of the file to read.
        
        Return:
            pandas dataframe containing the data.
        """
        try:
            return pd.read_table(infile)
        except FileNotFoundError:
            print("The file " + infile + " was not found")
            exit(1)
    
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
    
    def check_cols(self, f_name, df):
        fi = FileInfo()
        if f_name not in fi.check_cols.keys():
            return 
        else:
            for col in fi.check_cols[f_name]:
                if col not in df.columns.values:
                    print(col + " not in dataframe. Columns: "+str(df.columns))
                    continue                
                if not df[df[col] == 0][col].count() == len(df):                
                    print("WARNING: column "+col+" contains non-zero values")
    
    def remove_columns_and_write_files(self):
        """
        Drop extra columns from the dataframe and write cleaned data to file
        
        Parameters:
            - df, dataframe to remove columns from
            - drop_cols, list-like of strings, names of columns to drop
            - drop_if_non, list-like of strings, names of columns to drop iff
                these columns contain only zeros
        
        Return:
            pandas dataframe containing the merged data.
        """
        fi = FileInfo()
        
        print("\nFiles written to " + self.outpath + ":")
        for f in self.files:            
            if f in fi.is_csv:
                df = self.get_csv_data(self.inpath + f)
            else:
                df = self.get_tsv_data(self.inpath + f)
            
            if f not in fi.keep_cols.keys():
                print("*** WARNING: file "+f+" not in keep_cols dict ***")
            
            if fi.keep_cols[f] in df.columns.values:
                self.check_cols(f, df)
                df = df[['subreddit', fi.keep_cols[f]]].set_index('subreddit')
                df.to_csv(self.outpath + f)                
                print("\t" + f)
            else:
                print("*** WARNING: No "+fi.keep_cols[f]+" column in "+f+" ***")
                continue
            
## ****************************************************************************
## * MAIN METHOD
## ****************************************************************************
def main():
    """
    Main driver for program. Set filenames and parameters, then call MergeData.
    """
    
    raw_path = "../Data/Input/Raw/"
    processed_path = "../Data/Input/Processed/"
    
    cd = CleanData(raw_path, processed_path, 
                   files=["CountMediaEmbeds.txt"])
    cd.remove_columns_and_write_files()    
    
if __name__ == '__main__':
    main()

