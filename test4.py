from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, datetime
from scipy.signal import hilbert 
from scipy.fft import fft, ifft
import functools as ft 
from numpy_ext import rolling_apply
import sys
sys.setrecursionlimit(100000) # 10000 is an example, try with different values

df = pd.read_csv("data/apple.csv")

""" 
def snr(data):
    data['price'] = (data['high']+data['low'])/2
    data['H_L'] = data['high'] - data['low']
    #df['F'] = fft( np.array((df['high']+df['low'])/2))
    data['smooth'] = 0.0
    data['detrender'] = 0.0
    data['Q1'] = 0.0
    data['I1'] = 0.0
    data['jI'] = 0.0
    data['jQ'] = 0.0
    data['I2'] = 0.0
    data['Q2'] = 0.0
    data['Re'] = 0.0
    data['Im'] = 0.0
    data['Period'] = 0.0
    data['Period2'] = 0.0
    data['SmoothPeriod'] = 0.0
    data['Q3'] = 0.0
    data['I3'] = 0.0
    #data['Signal'] = 0.0
    #data['Noise'] = 0.0
    #data['SNR'] = 0.0

    data['smooth'] = data['price'].rolling(4).apply(lambda x: (4*x[3] + 3*x[2] + 2*x[1] + x[0])/10, raw=True)
    data['smooth'] = data['smooth'].fillna(0)
    
    data['detrender'] = ((.0962*data['smooth']) + (.5769*data['smooth'].shift(periods=2)) - (.5769*data['smooth'].shift(periods=4)) - (.0962*data['smooth'].shift(periods=6)))
    #data['detrender'] =  data['smooth'].rolling(7).apply(lambda x: (0.0962*x[6]) + (0.5769*x[4]) - (0.5769*x[2]) - (0.0962*x[0]), raw=True)
    data['detrender'] = data['detrender']*((.075*data['Period'].shift(periods=1)) +0.54)
                                        #((0.0962 * smooth) + (0.5769 * nz(smooth[2])) - (0.5769 * nz(smooth[4]))     -  (0.0962 * nz(smooth[6])))  * ((0.075 * nz(period[1])) + 0.54)
        

        #INPHASE AND QUADRATURE COMPONENTS 
    data['Q1'] =  data['detrender'].rolling(7).apply(lambda x: (0.0962*x[6]) + (0.5769*x[4]) - (0.5769*x[2]) - (0.0962*x[0]), raw=True)*((0.075*data['Period'].shift(periods=1)) + 0.54)
                            #((0.0962 * detrender)      + (0.5769 * nz(detrender[2]))     - (0.5769 * nz(detrender[4]))     - (0.0962 * nz(detrender[6])))    * ((0.075 * nz(period[1])) + 0.54)
    data['I1'] =  data['detrender'].shift(periods=3)
        
    for i in range(6, len(data)):    
        
        #ADVANCING PHASE BY 90º
        data['jI'][i] = ((0.0962*data['I1'][i]) + (0.5769*data['I1'][i-2]) - (0.5769*data['I1'][i-4]) - (.0962*data['I1'][i-6]))*((0.075*data['Period'][i-1]) + .54)
        data['jQ'][i] = ((0.0962*data['Q1'][i]) + (0.5769*data['Q1'][i-2]) - (0.5769*data['Q1'][i-4])- (.0962*data['Q1'][i-6]))*((.075*data['Period'][i-1]) + .54)

                            #((0.0962 * i1)   + (0.5769 * nz(i1[2]))  - (0.5769 * nz(i1[4]))  - (0.0962 * nz(i1[6]))) * ((0.075 * nz(period[1])) + 0.54)
        
        #PHASOR ADDITION FOR 3 BAR AVERAGING 
        data['I2'][i] = data['I1'][i] - data['jQ'][i]
        data['Q2'][i] = data['Q1'][i] + data['jI'][i]
        
        #SMOOTHING I AND Q COMPONENTS 
        data['I2'][i] = .2*data['I2'][i] + .8*data['I2'][i-1]
        data['Q2'][i] = .2*data['Q2'][i] + .8*data['Q2'][i-1]
        
        #HOMODYNE
        data['Re'][i] = data['I2'][i]*data['I2'][i-1] + data['Q2'][i]*data['Q2'][i-1]
        data['Im'][i] = data['I2'][i]*data['Q2'][i-1] - data['Q2'][i]*data['I2'][i-1]
 
        data['Re'][i] = .2*data['Re'][i] + .8*data['Re'][i-1]
        data['Im'][i] = .2*data['Im'][i] + .8*data['Im'][i-1] 
            
        if (data['Im'][i] != 0) and (data['Re'][i] != 0): data['Period'][i] = (2*np.pi)/np.arctan(data['Im'][i]/data['Re'][i])
        else: data['Period'][i] = 0
        #if data['Period'][i] > 1.5*data['Period'][i-1]: data['Period'][i] = 1.5*data['Period'][i-1]
        #if data['Period'][i] < .67*data['Period'][i-1]: data['Period'][i] = .67*data['Period'][i-1]
        #if data['Period'][i] < 6: data['Period'][i] = 6
        #if data['Period'][i] > 50: data['Period'][i] = 50

        data['Period'][i] = np.minimum(np.maximum(data['Period'][i], 0.67 * data['Period'][i-1]), 1.5 * data['Period'][i-1])
        data['Period'][i] = np.minimum(np.maximum(data['Period'][i], 6.0), 50.0)
        data['Period'][i] = (0.2 * data['Period'][i]) + (0.8 * data['Period'][i-1])


        #data['Period2'][i] = .2*data['Period'][i] + .8*data['Period'][i-1]
        data['SmoothPeriod'][i] = .33*data['Period'][i] + .67*data['SmoothPeriod'][i-1]

        #data['Q3'][i] = .5*(data['smooth'][i]-data['smooth'][i-2])*(.1759*data['SmoothPeriod'][i] + .4607)

        
        #for k in range(int(i/2)-1):
            #data['I3'][i] = data['I3'][i] + data['Q3'][i-k]
    
        #if int(data['SmoothPeriod'][i]/2) != 0:
            #data['I3'][i] = (1.57*data['I3'][i]) / int(data['SmoothPeriod'][i]/2)
        #data['Signal'][i] = data['I3'][i]**2 + data['Q3'][i]**2
        #data['Noise'][i] = (.1*data['H_L'][i]*data['H_L'][i]*.25) + .9*data['Noise'][i-1]
        #if data['Noise'][i] != 0 and data['Signal'][i] != 0:
            #data['SNR'][i] = .33*(10*np.log10(data['Signal'][i]/data['Noise'][i])) + .67*data['SNR'][i-1]
        
    #plt.plot(data['SNR'], color='red')
    #plt.show()
    return data
snr(df)  
 """



'''def instant_trend_line(data):
    #VARIABLES

    data['price'] = (data['high']+data['low'])/2
    data['smooth'] = 0.0
    data['detrender'] = 0.0
    data['I1'] = 0.0
    data['Q1'] = 0.0
    data['jI'] = 0.0
    data['jQ'] = 0.0
    data['I2'] = 0.0
    data['Q2'] = 0.0
    data['Re'] = 0.0
    data['Im'] = 0.0
    data['period'] = 0.0
    #data['Period2'] = 0.0
    data['SmoothPeriod'] = 0.0
    data['DCperiod'] = 0.0
    data['Itrend'] = 0.0
    data['smoothprice'] = 0.0
    data['trendline'] = 0.0
    
        #if data.isnull() == False:
    data['smooth'] = data['price'].rolling(4).apply(lambda x: (4*x[3] + 3*x[2] + 2*x[1] + x[0])/10, raw=True)
    #data['smooth'] = (4*data['price'] + 3*data['price'].shift(-1).shift(1) + 2*data['price'].shift(-2).shift(2) + data['price'].shift(-3).shift(3))/10
    data['detrender'] = data['smooth'].rolling(6).apply(lambda x: (0.0962*x[5] + 0.5769*x[4] - 0.5769*x[2] - 0.962*x[0])*(0.075*x[4] +0.54), raw=True)
    #data['detrender'] = (0.0962*data['smooth'] + 0.5769*data['smooth'].shift(-1).shift(1) - 0.5769*data['smooth'].shift(-3).shift(3) - 0.962*data['smooth'].shift(-5).shift(5))*(0.075*data['period'].shift(-1).shift(1) +0.54)

        #INPHASE AND QUADRATURE COMPONENTS 
    data['Q1'] = data['detrender'].rolling(7).apply(lambda x: (0.0962*x[6] + 0.5769*x[4] - 0.5769*x[2] - 0.0962*x[0])*(0.075*x[5] + 0.54), raw=True)
    #data['Q1'] = (0.0962*data['detrender'] + 0.5769*data['detrender'].shift(-2).shift(2) - 0.5769*data['detrender'].shift(-4).shift(4) - 0.0962*data['detrender'].shift(-6).shift(6))*(0.075*data['period'].shift(-1).shift(1) + 0.54)
    data['I1'] =  data['detrender'].rolling(4).apply(lambda x: x[0], raw=True)
        
        #ADVANCING PHASE BY 90º
    data['jI'] =data['I1'].rolling(7).apply(lambda x: (0.962*x[6] + 0.5769*x[4] - 0.5769*x[2] - 0.962*x[0])*(0.075*x[5] +0.54), raw=True) 
    #(.0962*data['I1'] + .5769*data['I1'].shift(-2).shift(2) - .5769*data['I1'].shift(-4).shift(4) - .0962*data['I1'].shift(-6).shift(6))*(.075*data['period'].shift(-1).shift(1) + .54)
    data['jQ'] = data['Q1'].rolling(7).apply(lambda x: (.0962*x[6] + .5769*x[4] - .5769*x[2] - .0962*x[0])*(.075*x[5] + .54), raw=True)
    #data['jQ'] = (.0962*data['Q1'] + .5769*data['Q1'].shift(-2).shift(2) - .5769*data['Q1'].shift(-4).shift(4) - .0962*data['Q1'].shift(-6).shift(6))*(.075*data['period'].shift(-1).shift(1) + .54)

        #PHASOR ADDITION FOR 3 BAR AVERAGING 
    data['I2'] = data['I1'] - data['jQ']
    data['Q2'] = data['Q1'] + data['jI']

        #SMOOTHING I AND Q COMPONENTS 
    data['I2'] = data['I2'].rolling(2).apply(lambda x: (2*x[1] + 8*x[0]), raw=True) 
    #.2*data['I2'] + .8*data['I2'].shift(-1).shift(1)
    data['Q2'] = data['Q2'].rolling(2).apply(lambda x: .2*x[1]+.8*x[0], raw=True)
    #data['Q2'] = .2*data['Q2'] + .8*data['Q2'].shift(-1).shift(1)

        #HOMODYNE
    data['Re'] = data['I2'].rolling(2).apply(lambda x: x[1]*x[0], raw=True) + data['Q2'].rolling(2).apply(lambda x: x[1]*x[0], raw=True)
    data['Im'] = data['I2'].rolling(2).apply(lambda x: x[1], raw=True)*data['Q2'].rolling(2).apply(lambda x: x[0], raw=True) - data['I2'].rolling(2).apply(lambda x: x[0], raw=True)*data['Q2'].rolling(2).apply(lambda x: x[1], raw=True)

    #data['Re'] = data.rolling(2).apply(lambda x: np.dot(x['I2'],x['I2'].shift(periods=-1)) + np.dot(x['Q2'],x['Q2'].shift(periods=-1)), raw=True)
    #data['Re'] = data['I2']*data['I2'].shift(-1).shift(1) + data['Q2']*data['Q2'].shift(-1).shift(1)
    #data['Im'] = data.rolling(2)apply(lambda x: np.dot(x['I2'],x['Q2'].shift(periods=-1)) - np.dot(x['Q2'],x['I2'].shift(periods=-1)), raw=True)
    #data['Im'] = data['I2']*data['Q2'].shift(-1).shift(1) - data['Q2']*data['I2'].shift(-1).shift(1)
    #data.fill_na(0)

    data['Re'] = data['Re'].rolling(2).apply(lambda x: .2*x[1] + .8*x[0], raw=True)
    #data['Re'] = .2*data['Re'] + .8*data['Re'].shift(-1).shift(1)
    data['Im'] = data['Im'].rolling(2).apply(lambda x: .2*x[1] + .8*x[0], raw=True)
    #data['Im'] = .2*data['Im'] + .8*data['Im'].shift(-1).shift(1)

    #data['x'] = data['Re'] * data['Im']

    data.loc[~((data['Im'] != 0) or (data['Re'] != 0)), 'period'] = (2*np.pi)/np.arctan(data['Im']/data['Re'])

    for i in range(1, len(data['price'])): 
    data.loc[data['Im']*data['Re'] != 0,  'period'] = (2*np.pi)/np.arctan(data['Im']/data['Re'])
        #if data['Im'][i] != 0 and data['Re'][i] != 0: data['period'][i] = (2*np.pi)/np.arctan(data['Im'][i]/data['Re'][i])
        #if data['x'][i] != 0: data['period'][i] = (2*np.pi)/np.arctan(data['Im'][i]/data['Re'][i])
        #data.loc[~((data['Im'][i] == 0) and (data['Re'][i] == 0)), 'period'] = (2*np.pi)/np.arctan(data['Im'][i]/data['Re'][i])
    #data.loc[data['period'] > 1.5*data['period'].shift(1),  'period']  = 1.5*data['period']  
        if data['period'][i] > 1.5*data['period'][i-1]: data['period'][i]= 1.5*data['period'][i-1]
    #data.loc[data['period'] < .67*data['period'].shift(1),  'period'] = .67*data['period']
        if data['period'][i] < .67*data['period'][i-1]: data['period'][i] = .67*data['period'][i-1]
    #data.loc[data['period'] < 6,  'period'] = 6 
        if data['period'][i] < 6: data['period'][i] = 6
    #data.loc[data['period'] > 50,  'period'] = 50
        if data['period'][i] > 50: data['period'][i] = 50

    data['period'] = data['period'].rolling(2).apply(lambda x: .2*x[1] + .8*x[0], raw=True)
    #data['period'] = .2*data['period'] + .8*data['period'].shift(-1).shift(1)
    data['SmoothPeriod'] = data['period'].rolling(2).apply(lambda x: .33*x[1] + .67*x[0], raw=True)
    #data['SmoothPeriod'] = .33*data['period'] + .67*data['SmoothPeriod'].shift(-1).shift(1)

        #COMPUTE TRENDLINE AS SIMPLE AVERAGE OVER THE DOMINANT CYCLE PERIOD 
    data['DCperiod'] = np.floor(data['SmoothPeriod'] + .5)
    #data['Itrend'] = 0.0
        #for k in range(i-1):
        #data['Itrend'][1] = data['price'][0] + data['Itrend'][0]
        #data['Itrend'][i] = data['price'][i-1] + data['Itrend'][i-1]
        #for k in range(2,len(data)-1):
            #data['Itrend'] = data['price'].rolling(window = k).sum()
        #for a in range(len(data)-1):
           # data['Itrend'][a+1] += data['Itrend'][a+1] + data['price'][a]

    #data['Itrend'] += data['price'].shift(-1).shift(1)
    #data['Itrend'] += data[['price','']].rolling(2).apply(lambda x: x[1]+x[0], raw=True)
    
    #car = ft.reduce(lambda a,b: a+b,data['price'])
    for i in range(len(data)-1):
        data['Itrend'][i] += data['price'][i]
        #data['Itrend'] = data['price'].rolling(len(data), min_periods=1).sum()

    data.loc[data['DCperiod'][i] > 0,  'Itrend'] = data['Itrend']/data['DCperiod']
    #if data['DCperiod'][i] > 0: data['Itrend'][i] = data['Itrend'][i]/data['DCperiod'][i]
        
    data['trendline'] = data['Itrend'].rolling(4).apply(lambda x: (4*x[3] + 3*x[2] + 2*x[1] + x[0])/10, raw=True)
    #data['trendline'] = (4*data['Itrend'] + 3*data['Itrend'].shift(-1).shift(1) + 2*data['Itrend'].shift(-2).shift(2) + data['Itrend'].shift(-3).shift(3))/10
    

    for i in range(len(data['price'])):    
        if i < 12: data['trendline'][i] = data['price'][i]

    data['smoothprice'] = (4*data['price'] + 3*data['price'].shift(-1).shift(1) + 2*data['price'].shift(-2).shift(2) + data['price'].shift(-3).shift(3))/10
    return data
instant_trend_line(df)'''



#data['FisherCG'] = .5 * np.log((1 + (1.98*(data['StochCG']-.5))) / (1 - (1.98*(data['StochCG']-.5)))) 
#data['FisherCG2'] = data['FisherCG'].shift(1)


pd.set_option('display.max_rows', None)
print(df)
df.to_csv('indicators.csv', na_rep='Unkown', float_format='%.2f')