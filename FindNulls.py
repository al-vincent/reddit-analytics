# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:44:22 2017

@author: Al
"""

from os import scandir
import pandas as pd

def get_filenames(path):
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

def main():
    path = "../Data/Input/Processed/SubscriberCounts/"
    f_names = get_filenames(path)
    f_out = path+"/Output/null_files.txt"

    print("\nNull subreddits:\n--------------")
    nulls = ["subreddit"]
    for f in f_names:
        df = pd.read_csv(path+f,index_col='subreddit',na_values='None')
        if df['mean_subscribers'].isnull().sum() > 0:
            df['null'] = df.isnull()            
            for value in df[df['null']].index.values:
                nulls.append(value)
                print(value)                
    
    pd.Series(nulls).to_csv(f_out, index=False)
        
if __name__ == '__main__':
    main()