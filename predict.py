import pandas as pd
from datetime import datetime
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

stock = pd.read_csv('sphist.csv')
stock["Date"] = pd.to_datetime(stock["Date"])
stock.sort_values(by=['Date'], ascending=False)

#create new columns on trends
def set_avg_price(col, days):
    new_col_name = col + "_" + str(days) +  "_avg" 
    stock[new_col_name] = stock[col].rolling(days).mean()
    stock[new_col_name] = np.roll(stock[new_col_name], -2)
    pos1 = len(stock) - 1 - days
    pos2 = len(stock) - 1
    stock.at[pos1:pos2 , new_col_name]=np.nan

def set_std_price(col, days):
    new_col_name = col + "_" + str(days) +  "_std" 
    stock[new_col_name] = stock[col].rolling(days).std()
    stock[new_col_name] = np.roll(stock[new_col_name], -2)
    pos1 = len(stock) - 1 - days
    pos2 = len(stock) - 1
    stock.at[pos1:pos2 , new_col_name]=np.nan


set_avg_price('Close', 5)
set_avg_price('Close', 30)
set_avg_price('Close', 365)
set_std_price('Close', 5)
set_std_price('Close', 365)
stock['ratio-5-365']= stock['Close_5_avg']/stock['Close_365_avg']

rows=stock[stock["Date"] > datetime(year=1951, month=1, day=2)]
stock.dropna(inplace=True)
 
#start creating models with new train and test
train=stock[stock["Date"] < datetime(year=2013,month=1,day=1)].copy()
test=stock[stock["Date"] >= datetime(year=2013,month=1,day=1)].copy()
train.drop(['High', 'Low', 'Open', 'Volume', 'Adj Close', 'Date'], axis=1, inplace=True)

cols=list(train.columns.values)
cols.remove('Close')

target = train['Close']
features = train[cols]
lr = LinearRegression()
lr.fit(features, target)
predictions = lr.predict(features)
mse_train = mean_squared_error(target, predictions)


target = test['Close']
features = test[cols]
lr = LinearRegression()
lr.fit(features, target)
predictions = lr.predict(features)
mse_test = mean_squared_error(target, predictions)

print('mse_test  ', mse_test)
print('mse_train ', mse_train)

