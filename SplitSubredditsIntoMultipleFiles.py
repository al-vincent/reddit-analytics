# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 07:26:11 2017

@author: Al
"""

import pandas as pd
from math import ceil

def main():
    # setup file paths
    f_in = "../Data/Input/Processed/SubredditLists/subreddits_1.txt"
    path = "../Data/Input/Processed/SubredditLists/SR_1/"
    
    # read in data; extract subreddit names 
    df = pd.read_csv(f_in)    
    subreddits = df['subreddit']
    
    # initialise vars for slicing
    NUM_FILES = 10
    start = 0; 
    increment = ceil(len(subreddits) / NUM_FILES)
    
    # slice out chunks of the list, write to file
    for i in range(NUM_FILES):
        f_out = path + "subreddits_1_" + str(i) + ".txt"
        end = (i + 1) * increment
        subreddits[start:end].to_csv(f_out, index=False,header=True)
        start = end

if __name__ == '__main__':
    main()