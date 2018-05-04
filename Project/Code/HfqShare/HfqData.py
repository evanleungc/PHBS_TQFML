#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:46:47 2018

@author: apple
"""

import pandas as pd
import os

class DataProcessing(object):
    '''
    Process High Frequency Data
    '''
    def __init__(self, path, length = None):
        self.path = path
        self.info = pd.DataFrame()
        self.length = length
        if length != None:
            self.length = length
        self.a_code = None
        self.trading_time = None
        self.stamp_dict = None
        self.name = None
        self.initialize()
        
    def initialize(self):
        self.__get_trading_time()
        self.__get_timestamp_dict()
            
    def get_a_code(self):
        '''
        Get A-Stock Code
        '''
        stock_code = os.listdir(self.path)
        a_code = []
        for i in stock_code:
            if i[2:5] == '000' and i[0:2] == 'SZ':
                a_code.append(i)
            elif i[2:5] == '002' and i[0:2] == 'SZ':
                a_code.append(i)
            elif i[2:4] == '30' and i[0:2] == 'SZ':
                a_code.append(i)
            elif i[2:4] == '60' and i[0:2] == 'SH':
                a_code.append(i)
            self.a_code = a_code
        return a_code
    
    def read_data(self, file, mode = 'Normal'):
        '''
        Read data and add headers
        '''
        info = pd.read_csv(self.path + file, header= None)
        if self.length != None and mode == 'Normal':
            info = info.iloc[0:self.length]
        info.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume', 'amount']
        info['inner_return'] = info['close'] / info['open'] - 1
        info['code'] = [file[2:-4]] * len(info)
        self.name = file[2:-4]
        return info
    
    def get_data(self, file):
        self.info = self.read_data(file)
        self.gen_time_stamp()
        return self.info
    
    def __get_trading_time(self):
        '''
        Get trading time
        '''
        szzs = self.read_data('SH000001.csv', mode = 'Special')
        trading_day = list(szzs['date'])
        trading_time = list(szzs['time'])
        trading_time = [i + '/' + j for i, j in zip(trading_day, trading_time)]
        unique_trading_time = list(pd.unique(trading_time))
        self.trading_time = unique_trading_time
        return unique_trading_time
    
    def __get_timestamp_dict(self):
        '''
        Generate timestamp based on actual trading time
        '''
        trading_time = self.trading_time
        stamp_list = list(range(len(trading_time)))
        stamp_dict = dict(zip(trading_time, stamp_list))
        self.stamp_dict = stamp_dict
    
    def gen_time_stamp(self):
        stamp_dict = self.stamp_dict
        self.info['uniquetime'] = [i + '/' + j for i, j in zip(self.info['date'],\
                        self.info['time'])]
        stamplist = []
        dellist = []
        uniquetimelist = []
        for idx, i in enumerate(self.info['uniquetime']):
            try:
                stamplist.append(stamp_dict[i])
            except:
                wrongend = int(i[-1])
                if wrongend < 5:
                    rightend = str(5)
                    i = i[0:-1] + rightend
                elif wrongend > 5:
                    rightend = str(int(i[-2]) + 1) + str(0)
                    if rightend == '60':
                        rightend = str(int(i[-5:-3]) + 1) + ':00'
                        i = i[0:-5] + rightend
                    else:
                        i = i[0:-2] + rightend
                try:
                    stamplist.append(stamp_dict[i])
                except:
                    print ('type2: %s' %idx)
                    print (self.info['code'].iloc[0])
                    print (self.info['uniquetime'].iloc[0])
                    stamplist.append(-999)
            uniquetimelist.append(i)
        self.info['uniquetime'] = uniquetimelist
        indexlist = list(self.info.index)
        for idx, i in enumerate(self.info['uniquetime']):
            if idx != 0 and stamp_dict[i] == stamp_dict[self.info['uniquetime'].iloc[idx - 1]]:
                dellist.append(indexlist[idx])
        self.info = self.info.drop(dellist, axis = 0)
        return self.info
                

    
    