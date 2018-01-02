# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 21:29:26 2017

@author: Al

Description: simple scraper / parser that gets numbers of subreddit subscribers from 
            http://redditmetrics.com/
             
            Takes a list of subreddits as input; uses urllib to navigate to the 
            appropriate redditmetrics page; finds the number of subscriber in the html; 
            extracts and parses these data; and averages to give #subscribers/month.
             
            Can be used in two different ways:
                1. Called from CLI, as ExtractRedditMetricsData.py <infile> <outfile>;
                2. With hard-coded file-paths.
             
            The ScrapeRedditMetrics class can also be used on its own to get metrics 
            for a single subreddit.
             
Notes:      1. Includes a wait-time between requests to redditmetrics.com, to avoid 
            DoS'ing the website. Currently set at 2 seconds. [** WARNING **: this is 
            good practice (IMO), but if data is required for lots of subreddits, the 
            script will take a *long* time to run!!]
            2. If the intention is to query a lot of subreddits, may be best to do it
            in small chunks (e.g. 100s, rather than 1000s or more).
"""

#-------------------------------------------------------------------------------------
# Import modules
#-------------------------------------------------------------------------------------
import urllib           # navigate to web page, get html data
from json import loads  # deserialise the html data for cleaning / parsing
import pandas as pd     # analysing and aggregating the data
from time import sleep  # add a pause to the scraping
import sys              # parsing cmd-line args, and exiting script cleanly

#-------------------------------------------------------------------------------------
# IMPORT MODULES
#-------------------------------------------------------------------------------------
class ScrapeRedditMetrics():
    def __init__(self, url):
        self.url = url  
        
    def get_script_text(self):        
        """ 
        Open the web page. Print warning (and continue) if page doesn't open. 
        Return the html from the page, or None if the page doesn't open.
        """
        html = None
        try: 
            page = urllib.request.urlopen(self.url).read()
            html = page.decode("utf8")
        except:
            print("*** WARNING: URL " + self.url + " did not open successfully. ***")
    
        return html

    def retrieve_data(self, html):
        """
        Find the part of the html with the total number of subscribers over time. 
        
        Parameters:
            - html, a single string containing all the html in the page.
        Return a string containing all the subscriber data, or None if 
            'total-subscribers' can't be found in the html for any reason
        """
        search_string = "element: 'total-subscribers',"
        
        # In the html, the subscriber info is an array of Javascript objects (or a list 
        # of python dicts), but extracted here as a single long string.
        start_segment = html.find(search_string)
        # make sure the search string exists
        if start_segment != -1:
            start_list = html.find("[", start_segment)
            end_list = html.find("]", start_list)
            return html[start_list:end_list + 1]
        else:
            return None

    def convert_text_to_dataframe(self, data_list):
        """
        Convert the string of subscriber data to a pandas dataframe (via JSON).         
        
        Parameters:
            - data_list, a string containing the total-subscribers JSON        
        Returns a pandas dataframe containing subscriber counts per day (as 
        a date object)
        """
        # clean up the fields
        data_list = data_list.replace("'", '"')
        data_list = data_list.replace('a', '\"subscriber_count\"')
        data_list = data_list.replace('y', '\"date\"')
        
        # convert the string to a list of python dicts
        try:
            subscriber_data = loads(data_list)
        except ValueError:
            print("*** WARNING: No data retrieved for "+self.url+" ***")
            return None
        
        # convert to dataframe and parse dates from string to 'date'
        df = pd.DataFrame(subscriber_data)
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
        
        return df
    
    def aggregate_output(self, subscriber_counts):
        """
        Take the daily subscriber values for the last day of each month, for the 12
        months of 2016 *only*, and average for these 12 values.
        
        NOTE: *a lot* of data is thrown away here! What's left is all I need; but 
        others may wish to alter this.
        
        Parameters:
            - subscriber_counts, a dataframe of subscriber counts per day 
        Returns a the mean subscriber value for 2016 (i.e. averaged over each month).
        """
        # get the last day of every month in the df
        mth_end_dates = subscriber_counts[subscriber_counts['date'].dt.is_month_end]
        # filter to only include the 12 dates from 2016
        mth_end_2016 = mth_end_dates[mth_end_dates['date'].dt.year == 2016]
        # return the average over these 12 months, i.e. a single value
        return mth_end_2016['subscriber_count'].mean()

    def scrape_and_parse(self):
        """
        Run all the methods above, to get the data, parse it an return the averaged 
        results. Return a dataframe with the averaged results if successful, or None 
        if unsuccessful.
        """
        # get the html
        text = self.get_script_text()
        # find the part that corresponds to total subscribers to the subreddit
        if text is not None:
            data_list = self.retrieve_data(text)
            # convert to a pandas dataframe
            if data_list is not None:
                # get monthly subscriber vals for 2016
                df = self.convert_text_to_dataframe(data_list)
                if df is not None:
                    return self.aggregate_output(df)
        return None

def get_subreddits(f_in):
    """
    Get a list of subreddits from an existing csv file.
    [NOTE: This is part of my wider workflow, where the data is held in .CSVs]
    
    Parameters:
        - f_in, a csv file containing data on subreddits (incl names of subreddits)
    Returns a list of subreddits (as strings)
    """
    try:
        df = pd.read_csv(f_in)
        return df['subreddit'].tolist()
    except FileNotFoundError:
        print("File " + f_in + " not found.")
        sys.exit(1)

def usage():
    print("Usage: python ExtractRedditMetricsData.py <input file> <output file>")
    sys.exit()

def main():
    """
    Main driver. Some general setup, including checks for how the script is being 
    run (see "Description" above).
    """
    
    # set the delay between requests to redditmetrics.com
    wait_time = 2
    
    # decide whether we're getting input and output filenames from the command line,
    # or whether they're hard-coded.
    if len(sys.argv) == 3:
        print("Using cmd-line filepaths")
        f_in = sys.argv[1]
        f_out = sys.argv[2]
    elif len(sys.argv) == 1:
        # setup file paths
        print("Using hard-coded filepaths")
        path = "../Data/Input/Processed/"
        f_in = path + "PostType.txt"
        f_out = path + "SubscriberCounts.txt"
    else:
        usage()
    
    # get list of subreddits
    subreddits = get_subreddits(f_in)
    # some subreddits for testing
#    subreddits = ["thedonald","AskReddit", "announcements"]    
    
    # set up the output file
    with open(f_out, 'w') as f:
        f.write("subreddit,mean_subscribers\n")

    # work through the list of subreddits, scraping the data for each one
    for subreddit in subreddits:
        url = "http://redditmetrics.com/r/"+subreddit 
        srm = ScrapeRedditMetrics(url)
        mean_subscribers = srm.scrape_and_parse()
        if mean_subscribers is not None:
            # write the result to the output file 
            # NOTE: I've found it's better to write after each subreddit is 
            # scraped (rather then wait until the list is processed), to avoid 
            # losing all data if an error is encountered.
            with open(f_out, 'a') as f:
                f.write(str(subreddit)+","+str(mean_subscribers)+"\n")
                
            print("Subreddit " + subreddit + " data retrieved successfully")
        
        # Pause before next request, to avoid DoS'ing the website
        sleep(wait_time)
    
if __name__ == '__main__':
    main()