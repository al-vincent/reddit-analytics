# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:23:41 2017

@author: Al
"""

import pandas as pd

class CleanData():
    def __init__(self, infile, outfile, start_time=0, min_subscribers=0):
        self.infile = infile
        self.outfile = outfile
        self.start_time = start_time
        self.min_subscribers = min_subscribers
        self.data = self.read_data() 
    
    def read_data(self):
        """
        Read data from infile (assumes .tsv-type format, which is consistent
        with MapReduce output).
        
        Return a pandas dataframe containing the data.
        """
        try:
            return pd.read_table(self.infile, index_col=0)
        except FileNotFoundError:
            print("The file " + self.infile + " was not found")
            exit(1)
    
    def remove_nulls(self):
        """
        Remove all data points where the number of subscribers to a subreddit 
        is -1 (null in the data), or 0 (i.e. there are no subscribers to the 
        subreddit)
        """
        self.data = self.data[self.data['subscriber_count'] > self.min_subscribers]
        
    def get_later_than(self):
        """
        Remove all dtaa points that occur before a pre-defined start-time (set
        using a Unix tiemstamp)
        """
        self.data = self.data[self.data['timestamp'] >= self.start_time]
    
    def write_clean_data(self):
        """
        Write the cleaned data to 'outfile' (as a csv)
        """
        self.data.to_csv(self.outfile)
    
    def clean_data(self):
        self.remove_nulls()
        self.get_later_than()
        self.write_clean_data()


def main():
    # input and output filepaths
    infile = "../Data/SubscriberCounts.txt"
    outfile = "../Data/SubscriberCountsClean.txt"
    
    # Unix timestamp for 1st Jan 2016
    jan_1st_2016 = 1451606400
    
    # Minimum number of subscribers (i.e. keep only subreddits with this many 
    # subscribers, or more)
    min_subscribers = 500
    
    # create a CleanData object
    cd = CleanData(infile, outfile, jan_1st_2016, min_subscribers)
    # clean the data and write to outfile
    cd.clean_data()    

if __name__ == '__main__':
    main()