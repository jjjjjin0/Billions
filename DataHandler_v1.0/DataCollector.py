import sqlalchemy
import psycopg2 as pg2
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import warnings
import time
warnings.simplefilter('ignore')

import talib as ta
from talib import MA_Type

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker

conn = pg2.connect(database='postgres', user='postgres', password='134308', host='175.124.190.196', port='5432')

# conn.autocommit = True
cur = conn.cursor()

sql = "select * from dailystockmarketpricedata where stockcode = '005930'"

print(datetime.now())
cur.execute(sql)
rows = cur.fetchall()
conn.commit()
print(datetime.now())
df = pd.DataFrame(rows, columns=['stockCode', 'date', 'Open', 'High', 'Low', 'Close', 'Volume',
                                 'Amount', 'adjPriceFactor', 'adjVolumeFactor', 'priceAdjustment'])

df['Open'] = pd.to_numeric(df['Open'])
df['High'] = pd.to_numeric(df['High'])
df['Low'] = pd.to_numeric(df['Low'])
df['Close'] = pd.to_numeric(df['Close'])
df['Volume'] = pd.to_numeric(df['Volume'])
df['Amount'] = pd.to_numeric(df['Amount'])

df['adjPriceFactor'] = df['adjPriceFactor'].astype(float)
df['adjVolumeFactor'] = df['adjVolumeFactor'].astype(float)

df = df.sort_values(by='date', ascending=False)
df = df.reset_index(drop=True)

df['Close shifted'] = df['Close'].shift(-1)
df['High shifted'] = df['High'].shift(-1)
df['Low shifted'] = df['Low'].shift(-1)

df['Upper BBand'], df['Middle BBand'], df['Lower BBand'] = ta.BBANDS(df['Close shifted'], timeperiod=20, )
df['RSI'] = ta.RSI(np.array(df['Close shifted']), timeperiod=14)
df['MACD'], df['MACD Signal'], df['MACD Hist'] = ta.MACD(df['Close shifted'], fastperiod=12,
                                                         slowperiod=26, signalperiod=9)
df['Momentum'] = ta.MOM(df['Close shifted'], timeperiod=12)
df['Returns'] = np.log(df['Close shifted']/df['Close shifted'].shift(-1))

df['Signal'] = 0
signalList = []
for ret in df['Returns']:

    if ret >= 0:
        signalList.append("1")
    else:
        signalList.append("0")

df['Signal'] = signalList

maxAbsScaler = preprocessing.MaxAbsScaler()

modelDict = {}

df.dropna(inplace=True)

X = np.array(df.drop(['Signal', 'Returns'], 1))
X = preprocessing.MaxAbsScaler.fit_transform(X)
Y = np.array(df['Signal'])

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3)

modelDict = {}
modelDict['X Train'] = xTrain
modelDict['X Test'] = xTest
modelDict['Y Train'] = yTrain
modelDict['Y Test'] = yTest

model = svm.SVC(kernel='rbf', decision_function_shape='ovo')

model.fit(modelDict['X Train'], modelDict['Y Train'])
yPredict = model.predict(modelDict['X Test'])

modelDict['Y Prediction'] = yPredict

# print("SVM Model Info for Ticker: "+i)
# print("Accuracy:",metrics.accuracy_score(Model_Dict[i]['Y Test'], Model_Dict[i]['Y Prediction']))
modelDict['Accuracy'] = metrics.accuracy_score(modelDict['Y Test'], modelDict['Y Prediction'])
modelDict['Precision'] = metrics.precision_score(modelDict['Y Test'], modelDict['Y Prediction'], pos_label=str(1), average="macro")
modelDict['Recall'] = metrics.recall_score(modelDict['Y Test'], modelDict['Y Prediction'], pos_label=str(1), average="macro")

prediction_length = len(modelDict['Y Prediction'])

df['SVM Signal'] = 0
df['SVM Returns'] = 0
df['Total Strat Returns'] = 0
df['Market Returns'] = 0

Signal_Column = df.columns.get_loc('SVM Signal')
Strat_Column = df.columns.get_loc('SVM Returns')
Return_Column = df.columns.get_loc('Total Strat Returns')
Market_Column = df.columns.get_loc('Market Returns')

df.iloc[-prediction_length:, Signal_Column] = list(map(int, modelDict['Y Prediction']))
df['SVM Returns'] = df['SVM Signal'] * df['Returns'].shift(-1)

df.iloc[-prediction_length:, Return_Column] = np.nancumsum(df['SVM Returns'][-prediction_length:])
df.iloc[-prediction_length:, Market_Column] = np.nancumsum(df['Returns'][-prediction_length:])

modelDict['Sharpe_Ratio'] = (df['Total Strat Returns'][-1] - df['Market Returns'][-1]) / np.nanstd(df['Total Strat Returns'][-prediction_length:])
