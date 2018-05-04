import numpy as np
import pandas as pd
import os
os.chdir('/Users/apple/Desktop/')
from HfqShare.Feature import Feature
path = '/Volumes/Seagate Backup Plus Drive/Stk_5F_2018/Stk_5F_2018/'
funcs = Feature(path)

def take_lag(df, colname, lagnum = 1):
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
    return np.array(retlist)

def take_lag2(df, colname, lagnum = 1):
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

def take_forward(df, colname, fnum = 1):
    templist = list(df[colname])
    retlist = templist[fnum:]
    retlist.extend([np.nan] * fnum)
    df['f_' + str(fnum) + '_' + colname] = retlist
    return df

hfqpath = '/Volumes/Seagate Backup Plus Drive/allticks/total/'
df1 = pd.read_csv(hfqpath + 'fivemindf_2018.csv')
df2 = pd.read_csv(hfqpath + 'fivemindf_2017.csv')
colname = list(df2.columns)
df1 = df1[colname]
df = pd.concat([df1,df2])

lag = 2

df = take_lag2(df, 'buyprice', lagnum = lag)
df = take_lag2(df, 'buyprice', lagnum = 1)
df = take_lag2(df, 'canbuy', lagnum = lag)
df['target'] = [0 for i in range(len(df))]
df['buyret'] = df['high'] / df['lag_%s_buyprice'%lag] - 1
df['risk'] = df['low'] / df['lag_%s_buyprice'%lag] - 1
#df['risk'][df['risk'] >= 0] =0.001
#df['risk'][df['risk'] < 0] = np.abs(df['risk'][df['risk'] < 0])
df['ret_risk_rate'] = df['buyret'] / df['risk']
df['target'][(df['buyret'] > 0.05) & (df['risk'] > -0.02) & (df['lag_%s_canbuy'%lag] == 1)] = 1
#df['target'][df['ret_risk_rate'] > 1] = 1
#df['target'][df['buyret'] < 0.02] = 0
df['buyret_next'] = df['high'] / df['lag_1_buyprice'] - 1
df['risk_next'] = df['low'] / df['lag_1_buyprice'] - 1


#hfqpath = '/Volumes/Seagate Backup Plus Drive/allticks/total/'
#df.to_csv(hfqpath + 'fivemindf_2018.csv', index = False)

hfqpath = '/Volumes/Seagate Backup Plus Drive/allticks/total/'

hfqamount = pd.DataFrame()
for i in os.listdir(hfqpath):
    if i[0:5] == 'group':
        hfqamount = hfqamount.append(pd.read_csv(hfqpath + i, index_col = 0))

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

lag = 2
hfqamount['code'] = hfqamount['code'].map(convert_stock)
hfqamount = hfqamount.drop('pchange_whole', axis = 1)
df['code'] = df['code'].map(convert_stock)
df['date'] = df['date'].map(convert_date)
mergedf = pd.merge(df, hfqamount, on = ['code', 'date'], how = 'left')
traindf = mergedf.dropna()
traindf = traindf.sort_values(['code', 'date'])
traindf = take_lag2(traindf, 'code')
traindf = traindf.sort_values(['code', 'date'])
traindf = traindf[traindf['code'] == traindf['lag_1_code']]
traindf = take_forward(traindf, 'code', fnum = lag)
traindf = traindf.sort_values(['code', 'date'])
traindf = traindf[traindf['code'] == traindf['f_%s_code'%lag]]
traindf = take_forward(traindf, 'target', fnum = lag)
traindf['f_%s_target'%lag][traindf['canbuy'] == 0] = np.nan
traindf = take_forward(traindf, 'buyret_next')
traindf = take_forward(traindf, 'risk_next')
traindf = traindf.sort_values(['code', 'date'])

days = 10
for i in range(1, days + 1):
    for j in range(8):
        traindf = take_lag2(traindf, 'buy_vol_%s'%j, i)
        traindf = take_lag2(traindf, 'sell_vol_%s'%j, i)
    traindf = take_lag2(traindf, 'sumbuy', i)
    traindf = take_lag2(traindf, 'sumsell', i)

for i in range(1, days + 1):
    traindf = take_lag2(traindf, 'code', lagnum = i)
for i in range(1, days + 1):
    traindf = traindf[traindf['code'] == traindf['lag_%s_code'%i]]

features = []
features.append('pchange')
for i in range(8):
    traindf['buy_rate_%s'%i] = traindf['buy_vol_%s'%i] / traindf['sumsell']
    features.append('buy_rate_%s'%i)
for i in range(8):
    traindf['sell_rate_%s'%i] = traindf['sell_vol_%s'%i] / traindf['sumbuy']
    features.append('sell_rate_%s'%i)
traindf['total_rate'] = traindf['sumbuy'] / traindf['sumsell']
features.append('total_rate')


for j in range(1, days + 1):
    for i in range(8):
        traindf['lag_%s_buy_rate_%s'%(j, i)] = traindf['lag_%s_buy_vol_%s'%(j, i)] / traindf['lag_%s_sumsell'%j]
        features.append('lag_%s_buy_rate_%s'%(j, i))
    for i in range(8):
        traindf['lag_%s_sell_rate_%s'%(j, i)] = traindf['lag_%s_sell_vol_%s'%(j, i)] / traindf['lag_%s_sumbuy'%j]
        features.append('lag_%s_sell_rate_%s'%(j, i))
    traindf['lag_%s_total_rate'%j] = traindf['lag_%s_sumbuy'%j] / traindf['lag_%s_sumsell'%j]
    features.append('lag_%s_total_rate'%j)
    traindf = take_lag2(traindf, 'pchange', j)
    features.append('lag_%s_pchange'%j)

dropidx = list(traindf[traindf['sumsell'] == 0].index)
traindf = traindf.drop(dropidx)
dropidx = list(traindf[traindf['sumbuy'] == 0].index)
traindf = traindf.drop(dropidx)

dropidx = list(traindf[traindf['total_rate'] > 5].index)
traindf = traindf.drop(dropidx)

def replace_inf(x):
    if np.isinf(x):
        y = 999
    else:
        y = x
    return y

traindf[features] = traindf[features].applymap(replace_inf)
traindf = traindf.dropna()
#traindf = traindf[(traindf['pchange'] > 0.01) & (traindf['pchange'] < 0.04)]

features.extend(['pchange', 'f_1_buyret_next', 'f_1_risk_next', 'code', 'date'])

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.optimizers
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

#traindf2.index = traindf2['date']
#traindf = traindf.drop(baddatelist)
#traindf = traindf2.loc[baddatelist]

X = np.array(traindf[features])
y = np.array(traindf['f_%s_target'%lag])
y = y.reshape((y.shape[0], 1))
x_train, x_test, y_train, y_test = train_test_split(X, y,\
                                                    test_size = 0.2, random_state = 11)

test_pchange = [i[-5] for i in x_test]
test_buyret = [i[-4] for i in x_test]
test_risk = [i[-3] for i in x_test]
test_code = [i[-2] for i in x_test]
test_date = [i[-1] for i in x_test]
x_train = np.array([i[0:-5] for i in x_train])
x_test = np.array([i[0:-5] for i in x_test])

sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

#from sklearn.decomposition import PCA
#pca = PCA(n_components=10)
#pca.fit(x_train)
#x_train = pca.transform(x_train)
#x_test = pca.transform(x_test)

import keras.backend as K
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

model = Sequential()
model.add(Dense(128, activation = 'relu', input_dim = x_train.shape[1]))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
portion = np.sum(traindf['target'] == 0) / np.sum(traindf['target'] == 1)
cw = {0: 1, 1: portion}
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=60, batch_size = 1024, class_weight=cw)
a = model.predict_classes(x_test)
d = model.predict(x_test)
b = [i[0] for i in a]
c = [i[0] for i in y_test]
d = [i[0] for i in d]
testdf = pd.DataFrame({'code': test_code, 'date': test_date, 'pred': b, 'prob': d, 'ret':test_buyret, 'risk':test_risk,'test': c,'pchange':test_pchange})
testdf = testdf.sort_values(['pred'], ascending = False)
a = testdf[testdf['pred'] == 1]
a = a.sort_values('prob', ascending = False)
#b = a[(a['pchange'] > 0.01) & (a['pchange'] < 0.04)]
#b = b[b['prob'] > 0.998]
def get_month(b, n = 10):
    b = b.iloc[0:n]
    b = b[b['prob'] > 0.9]
    if len(b) == 0:
        return [0, 0, 0, 0, 0]
    rate_1 = (np.sum((b['ret'] > 0.02) & (b['risk'] > -0.02)) / len(b))
    rate_2 = (np.sum(b['ret'] > 0.03) / len(b))
    rate_2_2 = (np.sum(b['ret'] > 0.05) / len(b))
    rate_3 = (np.sum(b['risk'] < -0.03) / len(b))
    rate_4 = (np.sum(b['risk'] < -0.05) / len(b))
    return [rate_1, rate_2, rate_2_2, rate_3, rate_4]

b = a[(a['pchange'] < 0.05) & (a['pchange'] > 0.03)]
groupdf = b.groupby('date').apply(get_month, (5))
grouprate = list(groupdf)

ratelist = np.array(grouprate[0])
count = 0
for i in grouprate[1:]:
    if i != [0, 0, 0, 0, 0]:
        count += 1
        ratelist = ratelist + np.array(i)
avg_rate = list(ratelist / count)
print (avg_rate)

baddatelist = []
for idx, i in enumerate(groupdf):
    if i[-2] > 0.3:
        baddatelist.append(groupdf.index[idx])



