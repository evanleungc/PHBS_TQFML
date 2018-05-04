#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 17:10:18 2018

@author: apple
"""

import pandas as pd
import numpy as np
import os
from HfqShare import HfqData as hd

class Feature(object):
        
    def __init__(self, path, codelist = None, reaction_period = 4):
        self.info = None
        self.path = path
        self.feature = []
        self.dateindex = []
        self.codeindex = []
        self.feature_name = []
        self.target = []
        self.reaction_period = reaction_period
        self.codelist = codelist
        self.dp = hd.DataProcessing(path)
        if self.codelist == None:
            dp = hd.DataProcessing(path)
            self.codelist = dp.get_a_code()
 
    def take_lag(self, df, colname, lagnum = 1):
        '''
        Get the lag 1
        
        Parameters
        ----------
        df: pandas groupby dataframe
        colname: str
            name of column to take lag
        lagnum: int
            lag number
        '''
        
        templist = list(df[colname])
        retlist = [np.nan] * lagnum
        retlist.extend(templist[:-lagnum])
        df['lag_' + str(lagnum) + '_' + colname] = retlist
        return df
    
    def map_feature(self, df, *funclist, paramlist = None):
        retlist = []
        if paramlist == None:
            for func in funclist:
                retlist.append(func(df))
        else:
            for idx, func in enumerate(funclist):
                retlist.append(func(df, *paramlist[idx]))
        return retlist
    
    def apply_feature(self, feature_name = None, intra_paramlist = None,\
                      daily_paramlist = None, **funclist):
        '''
        Apply multipy feature calculation to all data
        
        Parameters
        ----------
        *funclist: functions
            functions that are used to calculate features
        feature_name: list
            list of names to be declared
        param_dict: dict
            dict of parameters to be used in functions
        '''

        if 'intra' in funclist.keys():
            intra_length = len(funclist['intra'])
        else:
            intra_length = 0
        if 'daily' in funclist.keys():
            daily_length = len(funclist['daily'])
        else:
            daily_length = 0
        count = intra_length + daily_length
        if feature_name == None:
            feature_name = []
            for f in range(count):
                feature_name.append('f_%s'%f)
        featurelist = [[] for i in range(count)]
        feature_codelist = [[] for i in range(count)]
        feature_datelist = [[] for i in range(count)]
        for i in self.codelist:
            print (i[0:-4])
            self.info = self.dp.get_data(i)
            ##日内
            if intra_length != 0:
                if intra_paramlist == None:
                    feature = self.info.groupby(['date']).apply(self.map_feature, *funclist['intra'])
                else:
                    feature = self.info.groupby(['date']).apply(self.map_feature, \
                                               *funclist['intra'], paramlist = intra_paramlist)
                feature = feature.reset_index()
                for j in range(intra_length):
                    featurelist[j].extend(list([feature.iloc[:, 1].iloc[k][j] for k in range(feature.shape[0])]))
                    feature_codelist[j].extend([i[2:-4]] * len(feature))
                    feature_datelist[j].extend(list(feature['date']))
            ##每日
            if daily_length != 0:
                if daily_paramlist == None:
                    feature = self.map_feature(self.info, *funclist['daily'])
                else:
                    feature = self.map_feature(self.info, *funclist['daily'], paramlist = daily_paramlist)
                for j in range(daily_length):
                    featurelist[j + intra_length].extend(feature[j])
                    feature_codelist[j + intra_length].extend([i[2:-4]] * len(feature[j]))
                    feature_datelist[j + intra_length].extend(list(pd.unique(self.info['date'])))
                    
        for f in range(count):
            self.add_to_feature(featurelist[f], feature_codelist[f],\
                                feature_datelist[f], feature_name[f])            

    def gen_train_data(self, take_lag = True):
        dfdict = {}
        dfdict['code'] = self.codeindex[0]
        lag_codelist = [np.nan]
        lag_codelist.extend(self.codeindex[0][0:-1])
        dfdict['lag_1_code'] = lag_codelist
        dfdict['date'] = self.dateindex[0]
        for idx, i in enumerate(self.feature):
            if self.feature_name[idx] not in ['day_dist'] and take_lag == True:
                forward_feature = [np.nan]
                forward_feature.extend(i[0:-1])
                dfdict[self.feature_name[idx]] = forward_feature
            else:
                dfdict[self.feature_name[idx]] = i
        self.feature_df = pd.DataFrame(dfdict)
        return self.feature_df    
            
    def add_to_feature(self, featurelist, feature_codelist, feature_datelist, feature_name):
        '''
        Add feature to self.feature        
        '''
        self.feature.append(featurelist)
        self.codeindex.append(feature_codelist)
        self.dateindex.append(feature_datelist)
        self.feature_name.append(feature_name)
        