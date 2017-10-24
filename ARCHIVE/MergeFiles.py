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
from numpy import log
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sys import exit

## ****************************************************************************
## * DATA MAERGING CLASS
## ****************************************************************************
class MergeData():
    def __init__(self, f_comment, f_post, f_lang, f_self_link, f_contrib, 
                 f_urls, f_commentsPerPost, f_aveComments, f_out):
        """
        Constructor. Set up file references.
        
        Parameters:
            - f_comment, string, file containing comment-count data
            - f_post, string, file containing post-count data
            - f_lang, string, file containing Flesch-Kincaid readability data
            - f_self_link, string, file containing data on types of post
                (i.e. link posts vs self-posts).
            - f_contrib, string, file containing data on unique contributors
            - f_urls, string, file containing data on URL counts, averages
            - f_commentsPerPost, string, file containing data on mean, median
                and 90th %-ile comments-per-post
            - f_aveComments, string, file containing data on comment-length
            - f_out, string, file to write to 
        """
        self.f_comment = f_comment
        self.f_post = f_post
        self.f_lang = f_lang
        self.f_self_link = f_self_link
        self.f_contrib = f_contrib
        self.f_urls = f_urls
        self.f_commentsPerPost = f_commentsPerPost
        self.f_aveComments = f_aveComments
        self.f_out = f_out
    
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
      
    def merge_on_subreddits(self, df1, df2, attr='subreddit'):
        """
        Merges two dataframes on the attribute.
        
        Parameters:
            - df1, df2, dataframes to merge
            - attr, string, attribute to merge on
        
        Return:
            pandas dataframe containing the merged data.
        """
        if attr in df1.columns and attr in df2.columns:
            return df1.merge(df2, on=attr)
        else: 
            print("Error: cannot merge dataframes on attribute " + attr)
            exit(1)
    
    def remove_columns(self, df, drop_cols=None, drop_if_none=None):
        """
        Drop unnecessary columns from the dataframe.
        
        Parameters:
            - df, dataframe to remove columns from
            - drop_cols, list-like of strings, names of columns to drop
            - drop_if_non, list-like of strings, names of columns to drop iff
                these columns contain only zeros
        
        Return:
            pandas dataframe containing the merged data.
        """
        # drop cols from dataframe
        if drop_cols is not None: 
            if set(drop_cols).issubset(df.columns):
                df.drop(drop_cols, axis='columns', inplace=True)
            else:
                print("One or more of " + str(drop_cols) + 
                      " not in dataframe. Columns: " + str(df.columns))
        # drop cols from dataframe where all entries == 0
        if drop_if_none is not None:
            for lbl in drop_if_none:
                if lbl not in df.columns:
                    print(lbl + " not in dataframe. Columns: " + str(df.columns))
                    continue                
                if df[df[lbl] == 0][lbl].count() == len(df):
                    df.drop(lbl, axis='columns', inplace=True)
                else:
                    print("WARNING: column " + lbl + " contains non-zero values")
        
        return df

    def set_threshold(self, df, thresholds):
        """
        Filter the dataframe based on thresholds for particular attributes.
        
        Parameters:
            - df, dataframe to filter data from
            - thresholds, dict of form {"attribute" : filter_value} 
                    
        Return:
            pandas dataframe containing the filtered data.
        """        
        for param in thresholds:
            if param not in df.columns:
                print(param + " is not a column in the dataset. Columns: \n" + 
                      str(df.columns))
            else:
                df = df[df[param] >= thresholds[param]]
        
        return df
    
    def rescale_data(self, df):
        """
        Rescale data for specific attributes. Use either a log scale, or add a 
        small value to remove zeros before applying a log scale.
        
        Parameters:
            - df, dataframe to rescale
        
        Return:
            pandas dataframe containing the rescaled data.
        """
        
        SMALL_NUM = 0.01
        
        df['comment_count'] = log(df['comment_count'])
        df['post_count'] = log(df['post_count'])
        df['contributor_count'] = log(df['contributor_count'])
        df['average_num_urls'] = log(df['average_num_urls'] + SMALL_NUM)
        df['comments_per_post_mean'] = log(df['comments_per_post_mean'] + SMALL_NUM)
        df['average_character_count'] = log(df['average_character_count'] + SMALL_NUM)
        
        return df
    
    def apply_pca(self, df, num_pcs, use_whiten=False):
        """
        Apply Principal Components Analysis to dataframe.
        
        Parameters:
            - df, dataframe to apply PCA to
            - num_pcs, int, the number of principal componetns to use
            - use_whiten, boolean, flag to indicate whether data should be 
                whitened
        
        Return:
            pandas dataframe containing the transformed data.
        """
        # ensure data is scaled appropriately
        sub_reds = df.index
        df = StandardScaler().fit_transform(df)
        # create the PCA object with the desired number of principal components
        pca = PCA(n_components = num_pcs, whiten=use_whiten)
        # return the PCA'd data
        df = pd.DataFrame(pca.fit_transform(df))
         
        df.columns = ["PC_" + str(col) for col in df.columns.values]
        return df.set_index(sub_reds)
    
    def plot_distributions(self, df):
        """
        Plot distributions of attributes as histogram
        
        Parameters:
            - df, dataframe containing attributes to plot
        """
        for col in df.columns.values:
            plt.figure(); plt.hist(df[col], bins=200)
            plt.title("Histogram showing distribution of " + str(col))
            
    def get_ratio(self, df, col1, col2, new_col):
        """
        Get the ratio of two attributes and rename the resulting column.
        
        Parameters:
            - df, dataframe containing the attributes 
            - col1, string, the name of the numerator column of the ratio
            - col2, string, the name of the denominator column of the ratio
            - new_col, string, column name from the ratio values
        
        Return:
            pandas dataframe containing the data (incl new column).
        """
        if set([col1, col2]).issubset(df.columns):
            df[new_col] = df[col1] / df[col2]
        else:
            print("Error: one or more of " + 
                  str([col1, col2]) + " is not in dataframe.\nColumns in df: "+
                  df.columns.values)
        return df
    
    def print_params(self, thresholds, rescale, use_pca, whiten, save_file, 
                     f_out):
        """
        Print a series of parameters to console.
        
        Parameters:
            - thresholds, dict of form {"attribute" : filter_value}
            - rescale, boolean, indicator on whether data has been rescaled
            - use_pca, boolean, indicator on whether data has been PCA'd
            - whiten, boolean, indicator on whether data has been whitened
            - save_file, boolean, indicator ref saving data to file
            - f_out, string, output file                    
        """
        print("\nParameters Used\n---------------")
        if thresholds is not None:
            for t in thresholds:
                print('{:20s} {:7d}'.format(t+" min:", thresholds[t]))
        print('{:20s} {:>7}'.format("Rescale values:", str(rescale)))
        print('{:20s} {:>7}'.format("Use PCA:", str(use_pca)))
        print('{:20s} {:>7}'.format("Use PCA whitening:", str(whiten)))
        if save_file:
            print('{:20s} {}'.format("Output file:", f_out))
        else:
            print('{:20s} {:>7}'.format("Output file:", str(save_file)))
            
    def merge_data(self, drop_cols=None, drop_if_none=None, thresholds=None,
                   rescale=True, use_pca=True, pca_whiten=False,
                   plot_dists=False, show_params=True, write_file=True):
        """
        Main driver for class. Does the following:
        1. Reads data from several .csv and tsv files produced by MapReduce. 
        2. Mergesall files together in turn (inner join with subreddit as key).
        3. Removes unnecessary columns.
        4. Filter using thresholds; rescale; apply PCA; apply whitening; plot
            data distributions, print parameters to console, write data to file
            [ALL OPTIONAL]
        
        Parameters:
            - drop_cols, list-like of strings. Column names to drop.
            - drop_cols, list-like of strings. Columns to drop if vals all 0.
            - thresholds, dict of form {"attribute" : filter_value}
            - rescale, boolean, indicator on whether data has been rescaled
            - use_pca, boolean, indicator on whether data has been PCA'd
            - pca_whiten, boolean, indicator on whether data has been whitened
            - plot_dists, boolean, indicator on whether data should be plotted
            - show_params, boolean, indicator on whether data should be printed
            - write_file, boolean, indicator ref saving data to file
        """
        # load the data from the files
        comment_ct_data = self.get_tsv_data(self.f_comment)
        post_ct_data = self.get_tsv_data(self.f_post)
        language_data = self.get_tsv_data(self.f_lang)
        post_data = self.get_tsv_data(self.f_self_link)
        contributor_data = self.get_tsv_data(self.f_contrib)
        url_data = self.get_tsv_data(self.f_urls)
        commentsPerPostData = self.get_csv_data(self.f_commentsPerPost)
        aveCommentsData = self.get_tsv_data(self.f_aveComments)
        
        # merge the datasets (in pairs), using inner product
        df = self.merge_on_subreddits(comment_ct_data, post_ct_data)
        df = self.merge_on_subreddits(df, language_data)
        df = self.merge_on_subreddits(df, post_data)        
        df = self.merge_on_subreddits(df, contributor_data)
        df = self.merge_on_subreddits(df, url_data)
        df = self.merge_on_subreddits(df, commentsPerPostData)
        df = self.merge_on_subreddits(df, aveCommentsData)
        
        # calculate the ratio of median comments to post, to 90th %-ile
#        df = self.get_ratio(df, 'comments_per_post_median', 
#                            'comments_per_post_90th_percentile',
#                            'comments_per_post_ratio')
        
        # remove unnecessary columns
        df = self.remove_columns(df, drop_cols, drop_if_none)
        
        # apply thresholds (e.g. the min number of comments, posts for a
        # subreddit to be retained)
        if thresholds is not None: df = self.set_threshold(df, thresholds)            
        
        # apply log rescaling to the dataframe
        if rescale: df = self.rescale_data(df)
        
        # set the index to be the 'subreddit' column
        df.set_index("subreddit", inplace=True)        
        
        # perform PCA on the data
        if use_pca: df = self.apply_pca(df, len(df.columns.values), 
                                        use_whiten=pca_whiten)
        
        # plot histograms showing distributions of each attribute (if required)
        if plot_dists: self.plot_distributions(df)  
        
        # print the parameters used to the console
        if show_params: self.print_params(thresholds, rescale, use_pca, 
                                          pca_whiten, write_file, self.f_out)
        
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
    comment_file = "../Data/Input/CountComments.txt"
    post_file = "../Data/Input/CountPosts.txt"    
    language_file = "../Data/Input/LanguageComplexity.txt"
    self_link_file = "../Data/Input/PostType.txt"
    contributor_file = "../Data/Input/CountContributors.txt"
    urls_file = "../Data/Input/AverageURLs.txt"
    commentsPerPost_file = "../Data/Input/CommentsPerPost.txt"
    averageComment_file = "../Data/Input/AverageCommentCharacterCount.txt"
    
    outfile = "../Data/Output/MergedData_noPCA_noRescale.txt"
    
    parameter_thresholds = {'comment_count':100000, 'post_count':1000}
    
    md = MergeData(comment_file, 
                   post_file, 
                   language_file, 
                   self_link_file, 
                   contributor_file, 
                   urls_file, 
                   commentsPerPost_file, 
                   averageComment_file, 
                   outfile)
    
    md.merge_data(drop_cols=['post_id',
                             'num_link_posts',
                             'num_self_posts',
                             'urls_num_comments',
                             'num_comments',
                             'pc_self_posts',
                             'comment_ct',
                             'comments_per_post_median',
                             'comments_per_post_90th_percentile'], 
                  drop_if_none=['num_null_posts', 
                                'pc_null_posts'],
                  thresholds=parameter_thresholds, 
                  rescale=False,
                  use_pca=False,
                  pca_whiten=True,
                  plot_dists=False,
                  write_file=True)
    
if __name__ == '__main__':
    main()