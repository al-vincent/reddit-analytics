# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 07:15:03 2017

@author: Al
"""

import matplotlib.pyplot as plt
from datetime import date
import numpy as np

def convert_filename_to_date(f_name):
    year = int(f_name[3:7]) 
    month = int(f_name[8:10])
    # NOTE: arbitrarily assume that each data point is from 1st of the month
    return date.toordinal(date(year,month,1))

def main():
    # get data from file
    filename = "C:\\Users\\Al\\OneDrive\\Documents\\Royal Holloway MSc\\Project\\Reddit\\monthlyCount.txt"
    output = {'timestamp':[], 'count':[]}
    with open(filename, 'r') as f:
        #iterate through each line in the file
        for line in f:
            # split each line 
            file = line.split("  ")
            # TODO: convert file into timestamp
            pass    
            # store timestamp, count in output dict
            output['timestamp'].append(convert_filename_to_date(file[0]))
            output['count'].append(int(file[1].rstrip("\n")))
    # plot the output as time-series
    plt.scatter(output['timestamp'], output['count'])
    #plt.scatter(output['timestamp'], np.log(output['count']))
    #plt.scatter(np.log(output['timestamp']), output['count'])
    #plt.scatter(np.log(output['timestamp']), np.log(output['count']))

if __name__ == '__main__':
    main()