import sqlalchemy
import psycopg2 as pg2
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import warnings
import talib as ta

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


