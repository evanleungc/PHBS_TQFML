# Predict low-risk profitable trading opportunity with high frequency trading data
## Team members: <br />
* 梁康华 1601213555<br />
* 李君涵 1601213559<br />
* 纪雪云 1601213544<br />
* 王昊炜 1601213612
## 0 Structure
* [1.Motivation](#1-motivation)
* [2.Data Descriptions](#2-data-descriptions)
* [3.Feature Generation](#3-feature-generation)
* [4.Exploratory Data Analaysis](#4-exploratory-data-analaysis)
* [5.Training](#5-training) <br />
-[Logistic](#p_5.1) <br />
-[SVM](#p_5.2) <br />
-[DNN](#p_5.3) <br />
## 1 Motivation
* In the study of market microstructure, many researches prove that people with private information will buy and sell before non-informed traders.
* These traders tend to generate abnormal trading volume or price fluctuation in the market.
* High frequency data is a more precise and instant catch of these change and behavior.
* These behavior may generate complicated pattern and our team will attempt to employ machine learning algorithms to find these patterns and exploit profitable opportunity.
[<<<](#0-structure)
## 2 Data Descriptions
* The dataset we used in this project consists of two basic dataset:
* (1)**High Frequency Trading Volume Dataset**. It is collected by webspider from Sina Finance, which is a level-2 data consisting the 'Active Buy' or 'Active Sell' high frequency data
* (2)**5-min Frequency Trading Data**. It is collected in wind, which consists open, high, low ,close, trading volume and trading amount in 5-min frequency level
* We will combine these two basic dataset as the high frequency dataset we use in this project
* All A-Stocks in China are included[<<<](#0-structure)
## 3 Feature Generation
### 3.1. Generate 5-min Features
[Procedures in Code](https://github.com/evanleungc/PHBS_TQFML/blob/master/Project/Code/FeatureGeneration.ipynb) <br />
Variables Descriptions
* **buyprice:** It is the highest price in the last 20 miniutes of a trading day. We are going to assume we buy at this price.
* **canbuy:** If the stock has reached the price ceiling or floor, we assume that we cannot buy this stock and the variable will be 0. Otherwise, we assume that we can buy this stock and the variable will be 1. When doing the training, we will exclude all the samples with canbuy == 0.
* **buyret:** The return of buying at buyprice and sell in the highest price in the next two days.
* **risk:** The loss of buying at buyprice and sell in the lowest price in the next two days.
* **target:** The training target we use in this project. If in next twodays, the buyret > 0.03 and risk < -0.02, we consider it as a low-risk profitable trading opportunity and we set the label to be 1. Otherwise, we set it to be 0. 

 Within the variables, there are two features we construct from the 5-min frequency data <br />
* **amplitude:** This feature equals to (daily highest price/daily lowest price - 1), to measure the stock's variation. <br />
* **above_mean:** It is an indicator, which equals to 1 if closing price is higher than the mean price at closing time, and 0 otherwise. 
 ### 3.2. Generate High-Frequency Features
[Procedures in Code](https://github.com/evanleungc/PHBS_TQFML/blob/master/Project/Code/FeatureGeneration.ipynb) <br />
Variable Descriptions
* The high frequency volume will be divided in a **quintile** fashion based on the following thresholds: <br />
vollist = [0, 10000, 50000, 100000, 200000, 300000, 400000, 500000, infinity]
* The '0' in 'buy_rate_0' is the ratio of buying volume in [0, 10000] to total selling volume, showing the **small traders buying power** over total selling power. The relationship is as follow: <br />
'0': [0, 10000] <br />
'1': [10000, 50000] <br />
.<br/>
.<br/>
.<br/>
'6': [400000, 500000] <br />
'7': [500000, infinity] <br />
The **larger number** will show the **larger buying power** over total selling power.
* The **'sell\_rate_...'** is the opposite. It shows the selling power in a certain range over the total buying power.
* **'total\_rate'** indicate the total buying power over the total selling power
* **'pchange'** is the close price of the day / open price of the day - 1
* The **'lag_1**' means the feature one day before the trading day. We have 'lag_1', 'lag_2', 'lag_3' in this dataset.[<<<](#0-structure)
 ### 3.3 Conclusion on Features
* There are **74** features to train
* Features should be **standardized** before training because they are different in units
* **PCA** might be needed to handle the large amount of features[<<<](#0-structure)
 ## 4 Exploratory Data Analaysis
After data preprocessing, we can check whether there is any problem inside the datset
### 4.1. Missing Value Detection
* **No missing** values in the datasets after features generations
### 4.2. Imbalanced Check
* The imbalanced dataset problem is serious because **only 16% of the labels are 1** and the rest are 0.
### 4.3. Conclusion on Data
* Total observations: **347364**
* Timerange: **2017-09-04 --- 2018-02-28**
* **No missing data**
* **Serious imbalanced dataset**[<<<](#0-structure)
## 5 Training
### 5.1. Logistic Regression
[Procedures in Code](https://github.com/evanleungc/PHBS_TQFML/blob/master/Project/Code/logistic.ipynb)
#### 5.1.1. Feature Preprocessing and choose of  hyperparameters
* We train on **70%** of the sample and test on **30%** of the sample
* **SMOTE** transformation is used to tackle the 'imbalance dataset' problem
* To increase training speed, the data are **standardized**
* We use **PCA** to reduce dimension
#### 5.1.2. Hyperparameter Tunning
* To determine the **parameter C** in logistic regression, we use the **grid search**.
#### 5.1.3. Result
* The precision is 0.37, which is based on default threshold 50%
* However, if we raise the threshold, the precision should be quite high because the ROC is 0.8.<br />
![log-2](https://github.com/evanleungc/PHBS_TQFML/blob/master/Project/Report/photos/dnn-2.png)<br />
### 5.2. Decision Tree
[Procedures in Code](https://github.com/evanleungc/PHBS_TQFML/blob/master/Project/Code/Decision%20Tree.ipynb)
#### 5.2.1. Feature Preprocessing and choose of  hyperparameters
* We train on **70%** of the sample and test on **30%** of the sample
* **SMOTE** transformation is used to tackle the 'imbalance dataset' problem
* To increase training speed, the data are **standardized**
* We use **PCA** to reduce dimension
#### 5.2.2. Hyperparameter Tunning
* To determine the **parameter C** in logistic regression, we use the **grid search**.
#### 5.2.3. Result
* The precision is 0.30, which is based on default threshold 50%
* Even if we increase the threshold, the result is still not good <br />
![dtree-1](https://github.com/evanleungc/PHBS_TQFML/blob/master/Project/Report/photos/dtree-1.png)<br />
* The ROC shows that the performance is not as good as that of logistic regression<br />
![dtree-2](https://github.com/evanleungc/PHBS_TQFML/blob/master/Project/Report/photos/dtree-2.png)<br />
### 5.3. Deep Neural Network
[Procedures in Code](https://github.com/evanleungc/PHBS_TQFML/blob/master/Project/Code/DNN.ipynb)
#### 5.3.1. Feature Preprocessing
* **All features** are used in training
* To increase training speed, the data are **standardized**
* **'cw'** parameter in keras is used to tackle the 'imbalance dataset' problem
* We train on **70%** of the sample and test on **30%** of the sample
#### 5.3.2. DNN Structures after hyperparameter tunning
* We use **keras** package with tensorflow as kernel
* **Sequential** Model is used
* 1 input layer, 5 hidden layers, 1 output layers
* Input and all the hidden layers employ '**ReLu**' activation function
* The output layer employs '**Sigmoid**' activation function
* Parameter cw = {0: 1, 1: 5.32} indicates that we give more weights on '1' label because of the **imbalance dataset**
* The loss function we used in back propagation is '**binary_crossentropy**'
* **Adam optimizer** is used because it considers both momentum effect and avoids gradient exposure
#### 5.3.3. Model Test
When we are actually trading, we focus on whether we can profit from the model result. If the stock features predict '1', we will buy the stock and wait for profit. Therefore, '**Precision**' the right metric for us to evaluate the model.<br />
![dnn-1](https://github.com/evanleungc/PHBS_TQFML/blob/master/Project/Report/photos/dnn-1.png)<br />
The result is very encouraging.
We use the trained models to predict out-of-sample data.
The graph above shows that if we increase the thredsholds of predicting labels as 1, the precision increases gradually. We have **75% probability to succeed** if we buy stocks with **model prediction probabilities more than 90%**.<br />
![dnn-2](https://github.com/evanleungc/PHBS_TQFML/blob/master/Project/Report/photos/dnn-2.png)<br />
The ROC is 0.81, which is also another proof of the good result
