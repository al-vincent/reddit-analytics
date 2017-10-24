# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:36:10 2017

@author: A. Vincent
@description: Read data from text file and calculate summary statistics. Plot 
              simple graphs. 
              
              Data consists of number of comments made per subreddit per month, 
              and comes in the format:
                  <date>+space+<subreddit_name>+tab+<num_posts>
              where:
                  - <date> is a string in the format YYYY-MM
                  - <subreddit_name> is a single 'word' (i.e. alphanum string)
                  - <num_posts> is an integer
"""

## ****************************************************************************
## * IMPORT RELEVANT MODULES
## ****************************************************************************
import pandas as pd
import matplotlib.pyplot as plt
from sys import exit

## ****************************************************************************
## * DATA PROCESSING CLASS
## ****************************************************************************
class AnalyseSubreddits():    
    def __init__(self, infile, data_type):
        """
        Constructor. Set up instance variables and load data.
        
        Parameters:
            - infile, string, the input file containing the data.
            - data_type, string, indicates whether the data is from comments
                or posts
        """
        self.infile = infile
        self.d_type = data_type
        self.dates = ['Jan 2016', 'Feb 2016', 'Mar 2016', 'Apr 2016', 
                      'May 2016', 'Jun 2016', 'Jul 2016', 'Aug 2016', 
                      'Sep 2016', 'Oct 2016', 'Nov 2016', 'Dec 2016']
        self.data = self.load_data()
        self.top_n = self.data.groupby('subreddit')['count'].sum()

    def load_data(self):
        """
        Load data from tsv file. If file not found, print message to console
        and exit.
        """
        try:
            df = pd.read_table(self.infile, delim_whitespace=True, header=None, 
                               names=['date','subreddit','count'])
            return df
        except FileNotFoundError:
            print("File " + self.infile + " could not be found")
            exit(1)
        
    def get_dates(self):
        """
        Get a list of unique dates for data points
        """
        return self.data['date'].drop_duplicates()
    
    def single_subreddit_counts(self, subreddit):
        """
        Get the number of comments per month for single subreddit.
        
        Parameters:
            - subreddit, string, the subreddit to filter by
        
        Return:
            dataframe of sorted comment counts per month
        """
        df = self.data[self.data['subreddit'] == subreddit][['date','count']]
        return df.sort_values('date')
        
    def top_n_subreddits(self, n):
        """
        Get the top n subreddits in the dataset and the number of comments in 
        each one.
        
        Parameters:
            - n, int, the number of subreddits required
        
        Return:
            dataframe of n sorted comment counts 
        """
        return self.top_n.sort_values(ascending=False).head(n=n)
       
    def top_n_subreddits_per_month(self, n):
        """
        Get the top n subreddits in the dataset and the number of comments in 
        each one, per month
        
        Parameters:
            - n, int, the number of subreddits required
        
        Return:
            dataframe of n sorted comment counts per month
        """
        top_n = self.top_n_subreddits(n).reset_index()['subreddit'].tolist()
        return self.data[self.data['subreddit'].isin(top_n)].sort_values(['subreddit','date'])
                 
    def count_subreddits(self):
        """
        Return the counts of all comments, per subreddit, per month        
        """
        return self.data.groupby('date')['count'].sum()

    def plot_single_subreddit_counts(self, subreddit):
        """
        Plot a line graph showing the change in comment counts for a single
        subreddit.
        
        Parameters:
            - subreddit, string, the name of the subreddit
        """
        plt.figure()
        ttl = 'Number of '+self.d_type+' per month, '+subreddit+' subreddit'
        self.single_subreddit_counts(subreddit).plot.line(x='date',y='count',
                                    title=ttl)        
        plt.xticks(range(len(self.dates)), self.dates, rotation=45)        
        
    def plot_top_n_subreddits(self, n):
        """
        Plot a bar graph showing the n subreddits with the most comments across
        the whole dataset.
        
        Parameters:
            - n, int, number of subreddits to display.
        """
        df = self.top_n_subreddits(n)
        plt.figure(); 
        df.plot.bar(title = str(n) + ' subreddits with most ' + self.d_type)
    
    def plot_top_n_subbredits_per_month(self, n):
        """
        Plot a line graph showing the number of comments per month for the 
        n subreddits with the most comments across the whole dataset.
        
        Parameters:
            - n, int, number of subreddits to display.
        """
        top_n_pm = self.top_n_subreddits_per_month(n)
        top_n_pm.set_index('date', inplace=True)        
        df = top_n_pm.pivot(columns='subreddit')
                
        plt.figure(); 
        df.plot.line(title='Top '+str(n)+' subreddits per month by '+ 
                     self.d_type)
        plt.xticks(range(len(self.dates)), self.dates, rotation=45)
        
    def plot_count_subreddits(self):
        """
        Plot a line graph showing the total numebr of comments per month for 
        whole of Reddit.
        """
        ttl = 'Total number of '+self.d_type+' per month, all subreddits'
        plt.figure(); self.count_subreddits().plot.line(title=ttl)
        plt.xticks(range(len(self.dates)), self.dates, rotation=45)

## ****************************************************************************
## * HELPER FUNCTION
## ****************************************************************************
def print_output(header, data):
    """
    Pretty-print output to console
    """
    print('\n' + header + '\n' + '-' * len(header))
    print(data)

## ****************************************************************************
## * MAIN METHOD
## ****************************************************************************
def main():
    """
    Main driver function. 
    """
    # input file name
    infile = '../Data/Input/CommentCountsPerMonth.txt'
    
    # create 'analysis' object
    asr = AnalyseSubreddits(infile, 'comments')

    # plot subreddit counts
    print_output('Subreddit counts', asr.count_subreddits())
    asr.plot_count_subreddits()
    
    # plot top n subreddits
    n = 5
    print_output('Top '+str(n)+' subreddits in 2016', asr.top_n_subreddits(n))
    asr.plot_top_n_subreddits(n)
    
    # plot trend for single subreddit
    my_subred = 'The_Donald'#asr.top_n_subreddits(1).index.tolist()[0]
    print_output('<'+my_subred+'>, monthly comment counts', 
                 asr.single_subreddit_counts(my_subred))
    asr.plot_single_subreddit_counts(my_subred)
    
    # plot top n subreddits as monthly trends.
    print_output('Top '+str(n)+' subreddits in 2016, by month', 
                 asr.top_n_subreddits_per_month(n))
    asr.plot_top_n_subbredits_per_month(n)

if __name__ == '__main__':
    main()