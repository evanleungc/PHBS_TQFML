import numpy as np
import pandas as pd   
    
def get_canbuy(df):
    if np.mean(np.abs(df.iloc[-4:]['high'] - df.iloc[-4:]['low'])) == 0:
        return 0
    else:
        return 1

def growth_rate(df, lookback = 4):
    df = df.iloc[0 : len(df) - lookback]
    growth_r = df['close'].iloc[-1] / df['open'].iloc[0] - 1
    return growth_r

def amplitude(df, lookback = 4):
    df = df.iloc[0 : len(df) - lookback]
    ampl = np.max(df['high']) / np.min(df['low']) - 1 
    return ampl

def above_mean(df, lookback = 4):
    df = df.iloc[0 : len(df) - lookback]
    mean_line = np.mean(df['close'].iloc[0:-1])
    last_price = df['close'].iloc[-1]
    abmean = np.where(last_price>mean_line,1,0)
    return abmean
    
def get_high(df, lookback = None):
    if lookback == None:
        lookback = len(df)
    high = np.max(df['high'].iloc[0:lookback])
    return high

def get_low(df, lookback = None):
    if lookback == None:
        lookback = len(df)
    low = np.min(df['low'].iloc[0:lookback])
    return low

def get_close(df):
    last_close = df['close'].iloc[-1]
    return last_close

def get_buy(df):
    buy_price = np.max(df['high'].iloc[-4:])
    return buy_price

def take_lag(df, colname, lagnum = 1):
    '''
    Get the lag
    
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

def take_forward(df, colname, fnum = 1):
    templist = list(df[colname])
    retlist = templist[fnum:]
    retlist.extend([np.nan] * fnum)
    df['f_' + str(fnum) + '_' + colname] = retlist
    return df

def convert_stock(intcode):
    length = 6 - len(str(intcode))
    zeros = '0' * length
    strcode = zeros + str(intcode)
    return strcode

def convert_date(d):
    return d[0:4] + '-' + d[5:7] + '-' + d[8:]

def moving_sum(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N])

def get_sum_lag_vol(df, days):
    for i in range(8):
        mvsum = moving_sum(df['buy_vol_%s'%i].values, days)
        sumvol = [np.nan] * (days - 1)
        sumvol.extend(list(mvsum))
        df['buy_vol_%s'%i] = sumvol
        mvsum = moving_sum(df['sell_vol_%s'%i].values, days)
        sumvol = [np.nan] * (days - 1)
        sumvol.extend(list(mvsum))
        df['sell_vol_%s'%i] = sumvol
    mvsum = moving_sum(df['sumbuy'].values, days)
    sumvol = [np.nan] * (days - 1)
    sumvol.extend(list(mvsum))
    df['sumbuy'] = sumvol
    mvsum = moving_sum(df['sumsell'].values, days)
    sumvol = [np.nan] * (days - 1)
    sumvol.extend(list(mvsum))
    df['sumsell'] = sumvol
    for i in range(1, days + 1):
        df = take_lag2(df, 'code', lagnum = i)
    for i in range(1, days + 1):
        df = df[df['code'] == df['lag_%s_code'%i]]
    return df

def replace_inf(x):
    if np.isinf(x):
        y = 999
    else:
        y = x
    return y
