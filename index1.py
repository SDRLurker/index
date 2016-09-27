# -*- coding: utf-8 -*-
import pandas_datareader.data as web
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)

def make_train_data():
    start = datetime.datetime(2015,1,2)
    end = datetime.datetime(2016,7,22)
    f1 = web.DataReader("^DJI",'yahoo',start,end)
    f2 = web.DataReader("^KS11",'yahoo',start,end)
    dates = []
    x_data = []
    y_data = []
    for single_date in daterange(start, end):
        date_str = single_date.strftime('%Y-%m-%d')
        try:
                f1_close = f1.ix[date_str]['Close']
        except:
                continue
        try:
                f2_close = f2.ix[date_str]['Close']
        except:
                continue
        x_data.append(f1_close)
        y_data.append(f2_close)
        dates.append(date_str)
    return dates, x_data, y_data
                
def make_poly_x(pn, x_data):
    x_datas = []
    for n in range(pn+1):
        x_datas.append([ x ** n for x in x_data ])
    return x_datas
        
def print_hypothesis(pn, theta):
    print('h(X) = ')
    for i, element in enumerate(theta):
        if i == len(theta) - 1:
            print(element.item(0), " X ** ", i)
        else:
            print(element.item(0), " X ** ", i, " + ")

def get_hypothesis(pn, theta, x):
    y = 0
    for i, element in enumerate(theta):
        y += ( element.item(0) * ( x ** i) )
    return y
    
def get_sumerror(pn, theta, x_data, y_data):
    s = 0
    for i, x in enumerate(x_data):
      s += ( (y_data[i] - get_hypothesis(pn, theta, x)) ** 2)
    return s

# 프로그램 시작
PICKLE_FILE = 'data.pck'
import os.path
if os.path.isfile(PICKLE_FILE): 
    f = open('data.pck', 'rb')
    dates, x_data, y_data = pickle.load(f)
else:
    dates, x_data, y_data = make_train_data()
    f = open('data.pck', 'wb')
    pickle.dump([dates, x_data, y_data], f) 
f.close()

for i in range(len(dates)):
    print(dates[i], x_data[i], y_data[i])
    
# 정규방정식(normal equation) 알고리즘을 통해 cost 최소가 되는 가설함수를 찾습니다. 
print()
POLY = 1
x_datas = make_poly_x(POLY, x_data)
X = np.matrix(x_datas).transpose()
Xt = np.matrix(x_datas)
Y = np.matrix(y_data).transpose()
theta = np.dot(np.dot(np.linalg.pinv(np.dot(Xt, X)), Xt), Y)
print_hypothesis(POLY, theta)
print('get_sumerror : ', get_sumerror(POLY, theta, x_data, y_data))

# 다우존스지수가 18313.77일 때 코스피지수 값을 예측합니다.
print()
x_test = 18313.77
y_test = get_hypothesis(POLY, theta, x_test)
print('2016-08-02 DOW30 : ', x_test)
print('2016-08-02 estimated KOSPI : ', y_test)
print('2016-08-02 KOSPI : ', 2019.03)

# 그래프를 그립니다.
plt.plot(x_data, y_data, 'ro')
line_x = [ x for x in range( int(min(x_data)), int(max(x_data)) ) ]
line_y = [ get_hypothesis(POLY, theta, x) for x in range( int(min(x_data)), int(max(x_data)) ) ]
plt.plot(line_x, line_y)
plt.show()
