# -*- coding: utf-8 -*-
"""****************************************************************************
Created on Thu Sep  7 12:21:53 2017

@author:        A. VINCENT
Description:    Short script to be run on bigdata, to aggregate comments-per-
                post-per-subreddit, to average / median / 90th %-ile comments-
                per-subreddit.              
                
Notes:          a) The program is functional, but not well-tested or robust. 
                b) Input and output file paths are HARD-CODED(!!)
****************************************************************************"""

# import pandas library
import pandas as pd

def main():
    # set input and output files
    f_in = "path/to/input/file"
    f_out = "path/to/output/file"
    
    # read data
    df = pd.read_table(f_in)
    
    # calculate stats
    mn = df.groupby('subreddit')['count'].mean()
    median = df.groupby('subreddit')['count'].median()
    percentile = df.groupby('subreddit')['count'].quantile(q=0.9)
    stats=pd.DataFrame({'subreddit':mn.index.values,'mean':mn, 
                        'median':median, '90th_percentile':percentile})
    
    #print(stats.set_index('subreddit'))
    stats.set_index('subreddit').to_csv(f_out)
    
if __name__ == '__main__':
    main()