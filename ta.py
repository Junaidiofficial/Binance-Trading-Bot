import numpy as np 
import talib
import pandas as pd 
from numpy import genfromtxt
import csv, datetime


df = pd.read_csv("data/apple.csv", parse_dates =["date"], index_col ="date")

#retrives a csv file that contains our data and then converts it into a nympy list
#my_data = genfromtxt('15minutes.csv', delimiter=',')

print(my_data)

#form my_data the 4th index value contains the close
close = my_data[:,4]

print(close)


#close = numpy.random.random(100)


#simple moving avgerage tlib function (uses the close value of candle sticks, timperiod tiakes the first x values and then creates the average from them.)
#moving_average = talib.SMA(close, timeperiod=10)
#print(moving_average)

#rsi talib function (uses the close value of candle sticks, timperiod tiakes the first x values and then creates the average from them.)
rsi = talib.RSI(close, timeperiod=14)
print(rsi) 
