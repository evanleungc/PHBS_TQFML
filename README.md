# Predict low-risk profitable trading opportunity with high frequency trading data
## 1. Introduction
* In the study of market microstructure, many researches prove that people with private information will buy and sell before non-informed traders.
* These traders tend to generate abnormal trading volume or price fluctuation in the market.
* High frequency data is a more precise and instant catch of these change and behavior.
* These behavior may generate complicated pattern and our team will attempt to employ machine learning algorithms to find these patterns and exploit profitable opportunity.

## 2. Data Description
* traindf.csv is the dataset that has been preprocessed. It is generated from a high frequency trading volume dataset and 5-mins frequency price dataset.
* # of obs: 366310
* Time range: 2017-09-04 to 2018-02-28
* Variables Descriptions
'code': str, stock code
'date': str, date
'high': float, high price of the 5-min bar
'low' : float, low price of the 5-min bar
'buyprice': float, the highest price in the last 4 bar (20 mins) in the last trading day, which is the price I assume that I could have bought in the last trading day
'canbuy': int, 1--Indicate that I could buy in the last trading day (did not hit the price limitation); 0--Indicate the I could not buy in the last trading day (hit the price limitation)
'buyret': float, return I get if I buy with the 'buyprice'. Today's high / buyprice - 1
'risk': float, risk I bear if I buy with the 'buyprice'. Today's low / buyprice - 1
'target': int, 1--if buyret > 0.02 and risk > -0.01, 0--else
'sumsell': the sum of today's selling amount
'sumbuy': the sum of today's buying amount
sell_vol_[0...7]: separate selling amount into 8 groups based on scale
buy_vol_[0...7]: separate selling amount into 8 groups based on scale

## 3. Exploratory Data Analasis (EDA)
* Missing data analysis
* Distribution check
* Imbalance data check

## 4. Feature Engineering and Feature Preprocessing
* Transform raw features into economically meaningful features
* Preprocessing based on EDA and feature engineering
* PCA

## 5. Models Training
The Metric we use to evaluate will be "Precision" because we are interested in if we predict it to be a buying opportunity, how correctly we will actually get profit.
* Logistic regression
* SVM
* Xgboost
* Neural Network

## 6. Models Ensemble
* Bagging
