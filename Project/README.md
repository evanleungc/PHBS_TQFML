# Predict low-risk profitable trading opportunity with high frequency trading data
## Team members: <br />
* 梁康华 1601213555<br />
* 李君涵 1601213559<br />
* 纪雪云 1601213544<br />
* 王昊炜 1601213612
## 1 Motivation
* On one hand, in the study of market microstructure, many researches have proved that people with **private information** will buy and sell before non-informed traders.
* On the other hand, in China, **manipulating the market** does exist. People with **private information** tends to manipulate the market and earn money
* Combining the theory and the phenomenon in China, we would like to see whether we can make a profit based on them.
* Traders manipulating the market tend to generate **abnormal trading volume** or **price fluctuation** in the market.
* **High-frequency** data is a more precise and instant catch of these change and behavior.
* These behavior may generate **complicated pattern** and our team will attempt to employ **machine learning** algorithms to find these patterns and exploit profitable opportunity.

## 2. Data Description
* traindf.csv is the dataset that has been preprocessed. It is generated from a high frequency trading volume dataset and 5-mins frequency price dataset. [Example](http://vip.stock.finance.sina.com.cn/quotes_service/view/vMS_tradehistory.php?symbol=sh601208&date=2018-04-11)
* Numbers of obs: 347364
* Time range: 2017-09-04 to 2018-02-28
* Variables Descriptions
  * 'code': str, stock code
  * 'date': str, date
  * 'high': float, high price of the 5-min bar
  * 'low' : float, low price of the 5-min bar
  * **'buyprice'**: float, the highest price in the last 4 bar (20 mins) in the last trading day, which is the price I assume that I could have bought in the last trading day
  * 'canbuy': int, 1--Indicate that I could buy in the last trading day (did not hit the price limitation); 0--Indicate the I could not buy in the last trading day (hit the price limitation)
  * **'buyret'**: float, return I get if I buy with the 'buyprice'. Today's high / buyprice - 1
  * **'risk'**: float, risk I bear if I buy with the 'buyprice'. Today's low / buyprice - 1
  * **'target'**: int, 1--if buyret > 0.02 and risk > -0.01, 0--else
  * 'sumsell': float, the sum of today's selling amount
  * 'sumbuy': float, the sum of today's buying amount
  * sell_vol_[0...7]: float, separate selling amount into 8 groups based on scale
  * buy_vol_[0...7]: float, separate selling amount into 8 groups based on scale

## 3. Exploratory Data Analysis (EDA)
* Missing data analysis
* Imbalance data check

## 4. Feature Engineering and Feature Preprocessing
* Transform raw features into economically meaningful features
* Preprocessing based on EDA and feature engineering
* PCA

## 5. Models Training
The Metric we use to evaluate will be "Precision" because we are interested in if we predict it to be a buying opportunity, how correctly we will actually get profit.
* Logistic regression
* Decision Tree
* Deep Neural Network
