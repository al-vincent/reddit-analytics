# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 10:04:21 2017

@author: Al
"""

import pandas as pd
from sys import exit

# TODO:
# - [Create a list of tSNE input files?]
# - Look through each file in f_clusts
# - Read the data into a dataframe; store in a list / dict of dataframes 
#   [list / dict comp]
# - Merge the dataframes on 'subreddit', one at a time
# - Write the final dataframe to file

class CombineClusters:
    def __init__(self, f_tSNE, f_clust_dict, f_out, save_file):
        self.f_tSNE = f_tSNE
        self.f_clust_dict = f_clust_dict
        self.f_out = f_out
        self.save_file = save_file

    def load_file(self, f):
        try:
            return pd.read_csv(f)
        except FileNotFoundError:
            print("File " + f + " could not be found")
            exit(1)

    def load_many_files(self,f_dict):
        df_dict = {cl : self.load_file(f_dict[cl]) for cl in f_dict}        
        return df_dict

    def merge_files(self, data, df_dict, attr='subreddit'):
        for method in df_dict:
            data = data.merge(df_dict[method], on=attr)
            data.rename(columns = {'cluster' : method}, inplace=True)
        return data.set_index('subreddit')
    
    def write_file(self, df, f_out):
        if self.save_file:
            df.to_csv(f_out)
        else:
            print(df.head())
    
    def combine_clusters(self):
        tSNE = self.load_file(self.f_tSNE)
        df_list = self.load_many_files(self.f_clust_dict)
        df = self.merge_files(tSNE, df_list)
        self.write_file(df, self.f_out)
    
def main():
    f_tSNE = "../Data/Output/MergedData_PCA_whitened_tSNE.csv"
    f_clusts = {"ahc_d13" : "../Data/Output/h_clusters_d13.csv",
                "dpgmm_c40" : "../Data/Output/dpgmm_clusters_c40.csv"}
    f_out = "../Data/Output/tsne_clusters.csv"
    
    cc = CombineClusters(f_tSNE, f_clusts, f_out, save_file=True)
    cc.combine_clusters()

if __name__ == '__main__':
    main()