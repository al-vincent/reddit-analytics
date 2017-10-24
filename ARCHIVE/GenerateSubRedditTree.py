# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:57:56 2017

@author: Al
"""

import json
from sys import exit

class ExtractFeatures:
    def __init__(self, infile):
        self.infile = infile
        self.features = dict()

    def load_parse_data(self):
        try:
            with open(self.infile, 'r') as f:
                for line in f:
                    j_line = json.loads(line)
                    self.extract_features(j_line)
        except FileNotFoundError: 
            print("File " + self.infile + " could not be found")
            exit(1)
    
    def extract_features(self, json_line):
           
        if json_line['parent_id'] not in self.features.keys():
            self.features[json_line['parent_id']] = [{json_line['name'] : int(json_line['created_utc'])}]
        else:
            self.features[json_line['parent_id']].append({json_line['name'] : int(json_line['created_utc'])})                        

    def get_feature_dict(self):
        return self.features

def main():
    infile = '..\\Data\\SampleData\\RC_2014-01_1000lines.json'
        
    ef = ExtractFeatures(infile)
    ef.load_parse_data()
    d = ef.get_feature_dict()
    for key in d:
        if len(d[key]) > 1:
            print('Key: '+key+', values: '+str(d[key]))           

if __name__ == '__main__':
    main()