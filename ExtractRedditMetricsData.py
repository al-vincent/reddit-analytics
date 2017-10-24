# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 21:29:26 2017

@author: Al
"""

import urllib
#from bs4 import BeautifulSoup
from json import loads
import pandas as pd
from time import sleep
import sys 

class ScrapeRedditMetrics():
    def __init__(self, url, subreddit):
        self.url = url
        self.subreddit = subreddit
        
    def get_script_text(self):        
        # open the page
        html = None
        try: 
            page = urllib.request.urlopen(self.url).read()
            html = page.decode("utf8")
        except:
            print("*** WARNING: URL " + self.url + " did not open successfully. ***")
# =============================================================================
#         # get the html
#         soup = BeautifulSoup(page, 'html.parser')
#         
#         # find all the <script> tags
#         script_txt = soup.find_all("script")
# 
#         return script_txt[8].string
# =============================================================================
    
        return html

    def retrieve_data(self, html):
        search_string = "element: 'total-subscribers',"
        
        start_segment = html.find(search_string)
        if start_segment != -1:
            start_list = html.find("[", start_segment)
            end_list = html.find("]", start_list)
            return html[start_list:end_list + 1]
        else:
            return None

    def convert_text_to_dataframe(self, data_list):
        # clean up the fields
        data_list = data_list.replace("'", '"')
        data_list = data_list.replace('a', '\"subscriber_count\"')
        data_list = data_list.replace('y', '\"date\"')
        
        try:
            subscriber_data = loads(data_list)
        except ValueError:
            print("*** WARNING: No data retrieved for "+self.url+" ***")
            return None
        
        df = pd.DataFrame(subscriber_data)
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
        
        return df
    
    def aggregate_output(self, subscriber_counts):
        mth_end_dates = subscriber_counts[subscriber_counts['date'].dt.is_month_end]
        mth_end_2016 = mth_end_dates[mth_end_dates['date'].dt.year == 2016]
        return mth_end_2016['subscriber_count'].mean()

    def scrape_and_parse(self):
        text = self.get_script_text()
        if text is not None:
            data_list = self.retrieve_data(text)
            if data_list is not None:
                df = self.convert_text_to_dataframe(data_list)
                if df is not None:
                    return self.aggregate_output(df)
        return None

def get_reddits(f_in):
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
    
    if len(sys.argv) == 3:
        print("Using cmd-line filepaths")
        f_in = sys.argv[1]
        f_out = sys.argv[2]
    elif len(sys.argv) == 1:
        # setup file paths
        print("Using hard-coded filepaths")
        path = "../Data/Input/Processed/SubscriberCounts/"
        f_in = path + "Output/null_files.txt"
        f_out = path + "SubscriberCounts09_replacements2.txt"
    else:
        usage()
    
    # get list of subreddits
    subreddits = get_reddits(f_in)
    
    # set the url for the page to scrape    
#    urls = ["http://redditmetrics.com/r/thedonald",
#            "http://redditmetrics.com/r/AskReddit"]
    with open(f_out, 'w') as f:
        f.write("subreddit,mean_subscribers\n")
    #subscriber_counts = []
    #for subreddit in subreddits[:20]:
    for subreddit in subreddits:
        url = "http://redditmetrics.com/r/"+subreddit 
        srm = ScrapeRedditMetrics(url, subreddit)
        mean_subscribers = srm.scrape_and_parse()
        if mean_subscribers is not None:
#            subscriber_counts.append({"subreddit":subreddit,
#                                     "mean_subscribers":srm.scrape_and_parse()})
            with open(f_out, 'a') as f:
                f.write(str(subreddit)+","+str(mean_subscribers)+"\n")
                
            print("Subreddit " + subreddit + " data retrieved sucessfully")
            
        sleep(2)
    #print(subscriber_counts)
    
    #print(pd.DataFrame(subscriber_counts).set_index('subreddit'))
#    mean_subscribers = pd.DataFrame(subscriber_counts).set_index('subreddit')
#    mean_subscribers.to_csv(f_out)
    
if __name__ == '__main__':
    main()