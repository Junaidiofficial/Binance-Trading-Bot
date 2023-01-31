from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, datetime
import math
from scipy.signal import hilbert 
from scipy.fft import fft, ifft
import functools as ft 
from numpy_ext import rolling_apply
import sys
sys.setrecursionlimit(100000) # 10000 is an example, try with different values

df = pd.read_csv("data/apple.csv")
""" 
def sine(data, alpha=.07):
    data['price'] = (data['high']+data['low'])/2
    #df['F'] = fft( np.array((df['high']+df['low'])/2))
    data['smooth'] = 0.0
    data['Cycle'] = 0.0
    data['detrender'] = 0.0
    data['Q1'] = 0.0
    data['I1'] = 0.0
    data['DeltaPhase'] = 0.0
    data['MedianPhase'] = 0.0
    data['jI'] = 0.0
    data['jQ'] = 0.0
    data['I'] = 0.0
    data['Q'] = 0.0
    data['I2'] = 0.0
    data['Q2'] = 0.0
    data['1Re'] = 0.0
    data['1Im'] = 0.0
    data['Re'] = 0.0
    data['Im'] = 0.0
    data['Period'] = 0.0
    data['Period2'] = 0.0
    data['SmoothPeriod'] = 0.0
    data['SmoothPrice'] = 0.0
    data['RealPart'] = 0.0
    data['ImagPart'] = 0.0
    data['DCPhase'] = 0.0
    data['DCPeriod'] = 0.0
    data['Sine'] = 0.0
    data['Sine_45'] = 0.0
    for i in range(6, len(data['price'])):
        
        data['smooth'][i] = (4*data['price'][i] + 3*data['price'][i-1] + 2*data['price'][i-2] + data['price'][i-3])/10
        data['detrender'][i] = (0.0962*data['smooth'][i] + 0.5769*data['smooth'][i-1] - 0.5769*data['smooth'][i-3] - 0.962*data['smooth'][i-5])*(0.075*data['Period'][i-1] +0.54)

        #INPHASE AND QUADRATURE COMPONENTS 
        data['Q1'][i] = (0.0962*data['detrender'][i] + 0.5769*data['detrender'][i-2] - 0.5769*data['detrender'][i-4] - 0.0962*data['detrender'][i-6])*(0.075*data['Period'][i-1] + 0.54)
        data['I1'][i] =  data['detrender'][i-3]
        
        #ADVANCING PHASE BY 90º
        data['jI'][i] = (.0962*data['I1'][i] + .5769*data['I1'][i-2] - .5769*data['I1'][i-4] - .0962*data['I1'][i-6])*(.075*data['Period'][i-1] + .54)
        data['jQ'][i] = (.0962*data['Q1'][i] + .5769*data['Q1'][i-2] - .5769*data['Q1'][i-4] - .0962*data['Q1'][i-6])*(.075*data['Period'][i-1] + .54)

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
            
        if data['Im'][i] != 0 and data['Re'][i] != 0: data['Period'][i] = (2*np.pi)/np.arctan(data['Im'][i]/data['Re'][i])
        if data['Period'][i] > 1.5*data['Period'][i-1]: data['Period'][i] = 1.5*data['Period'][i-1]
        if data['Period'][i] < .67*data['Period'][i-1]: data['Period'][i] = .67*data['Period'][i-1]
        if data['Period'][i] < 6: data['Period'][i] = 6
        if data['Period'][i] > 50: data['Period'][i] = 50

        data['Period2'][i] = .2*data['Period'][i] + .8*data['Period'][i-1]
        data['SmoothPeriod'][i] = .33*data['Period2'][i] + .67*data['SmoothPeriod'][i-1]

        #compute dominant cycle

        data['SmoothPrice'][i] = (4*data['price'][i] + 3*data['price'][i-1] + 2*data['price'][i-2] + data['price'][i-3])/10
        data['DCPeriod'][i] = (data['SmoothPrice'][i]+ .5).astype(int) 

        for k in range(i-1):
            data['RealPart'][i] = data['RealPart'][i] + (np.sin(np.deg2rad(360*k / data['DCPeriod'][i])) * data['SmoothPrice'][i-k])
            data['ImagPart'][i] = data['ImagPart'][i] + (np.cos(np.deg2rad(360*k / data['DCPeriod'][i])) * data['SmoothPrice'][i-k])
            
    
        if np.abs(data['RealPart'][i]) > 0.001: data['DCPhase'][i] = 180/np.pi * np.arctan(data['ImagPart'][i]/data['RealPart'][i])
        if np.abs(data['RealPart'][i]) <= 0.001:  data['DCPhase'][i] = 90 * np.sin(np.deg2rad(data['ImagPart'][i]))
        
        data['DCPhase'][i] = data['DCPhase'][i] + 90
        
        #compensate for 1 bar of lage of the WMA
        data['DCPhase'][i] = data['DCPhase'][i] + 360 / data['SmoothPeriod'][i]
        
        if data['ImagPart'][i] < 0: data['DCPhase'][i] = data['DCPhase'][i] + 180
        if data['DCPhase'][i] > 315: data['DCPhase'][i] = data['DCPhase'][i] - 360
        
        data['Sine'][i] = np.sin(np.deg2rad(data['DCPhase'][i]))
        data['Sine_45'][i] = np.sin(np.deg2rad(data['DCPhase'][i] + 45))
        
    return data
sine(df)  """
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
    data['I'] = 0.0
    data['Q'] = 0.0
    data['I2'] = 0.0
    data['Q2'] = 0.0
    data['1Re'] = 0.0
    data['1Im'] = 0.0
    data['Re'] = 0.0
    data['Im'] = 0.0
    data['Period'] = 0.0
    data['Period2'] = 0.0
    data['SmoothPeriod'] = 0.0
    data['Q3'] = 0.0
    data['I3'] = 0.0
    data['Signal'] = 0.0
    data['Noise'] = 0.0
    data['SNR'] = 0.0
    for i in range(6, len(data['price'])):
        
        data['smooth'][i] = (4*data['price'][i] + 3*data['price'][i-1] + 2*data['price'][i-2] + data['price'][i-3])/10
        data['detrender'][i] = (0.0962*data['smooth'][i] + 0.5769*data['smooth'][i-1] - 0.5769*data['smooth'][i-3] - 0.962*data['smooth'][i-5])*(0.075*data['Period'][i-1] +0.54)

        #INPHASE AND QUADRATURE COMPONENTS 
        data['Q1'][i] = (0.0962*data['detrender'][i] + 0.5769*data['detrender'][i-2] - 0.5769*data['detrender'][i-4] - 0.0962*data['detrender'][i-6])*(0.075*data['Period'][i-1] + 0.54)
        data['I1'][i] =  data['detrender'][i-3]
        
        #ADVANCING PHASE BY 90º
        data['jI'][i] = (.0962*data['I1'][i] + .5769*data['I1'][i-2] - .5769*data['I1'][i-4] - .0962*data['I1'][i-6])*(.075*data['Period'][i-1] + .54)
        data['jQ'][i] = (.0962*data['Q1'][i] + .5769*data['Q1'][i-2] - .5769*data['Q1'][i-4] - .0962*data['Q1'][i-6])*(.075*data['Period'][i-1] + .54)

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
            
        if data['Im'][i] != 0 and data['Re'][i] != 0: data['Period'][i] = (2*np.pi)/np.arctan(data['Im'][i]/data['Re'][i])
        if data['Period'][i] > 1.5*data['Period'][i-1]: data['Period'][i] = 1.5*data['Period'][i-1]
        if data['Period'][i] < .67*data['Period'][i-1]: data['Period'][i] = .67*data['Period'][i-1]
        if data['Period'][i] < 6: data['Period'][i] = 6
        if data['Period'][i] > 50: data['Period'][i] = 50

        data['Period2'][i] = .2*data['Period'][i] + .8*data['Period'][i-1]
        data['SmoothPeriod'][i] = .33*data['Period2'][i] + .67*data['SmoothPeriod'][i-1]

        data['Q3'][i] = .5*(data['smooth'][i]-data['smooth'][i-2])*(.1759*data['SmoothPeriod'][i] + .4607)
        
        for k in range(int(i/2)-1):
            data['I3'][i] = data['I3'][i] + data['Q3'][i-k]
    
        if int(data['SmoothPeriod'][i]/2) != 0:
            data['I3'][i] = (1.57*data['I3'][i]) / int(data['SmoothPeriod'][i]/2)
        data['Signal'][i] = data['I3'][i]**2 + data['Q3'][i]**2
        data['Noise'][i] = (.1*data['H_L'][i]*data['H_L'][i]*.25) + .9*data['Noise'][i-1]
        if data['Noise'][i] != 0 and data['Signal'][i] != 0:
            data['SNR'][i] = .33*(10*np.log10(data['Signal'][i]/data['Noise'][i])) + .67*data['SNR'][i-1]
        
    #plt.plot(data['SNR'], color='red')
    #plt.show()
    return data
snr(df)  """

""" 
def instant_trend_line(data, period = 14):
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
    data['smoothperiod'] = 0.0
    data['DCperiod'] = 0.0
    data['realpart'] = 0.0 
    data['imagpart'] = 0.0
    data['count'] = 0.0
    data['Itrend'] = 0.0
    data['smoothprice'] = 0.0
    data['trendline'] = 0.0

    for i in range(6, len(data['price'])):
        
        data['smooth'][i] = (4*data['price'][i] + 3*data['price'][i-1] + 2*data['price'][i-2] + data['price'][i-3])/10
        data['detrender'][i] = (0.0962*data['smooth'][i] + 0.5769*data['smooth'][i-1] - 0.5769*data['smooth'][i-3] - 0.962*data['smooth'][i-5])*(0.075*data['period'][i-1] +0.54)

        #INPHASE AND QUADRATURE COMPONENTS 
        data['Q1'][i] = (0.0962*data['detrender'][i] + 0.5769*data['detrender'][i-2] - 0.5769*data['detrender'][i-4] - 0.0962*data['detrender'][i-6])*(0.075*data['period'][i-1] + 0.54)
        data['I1'][i] =  data['detrender'][i-3]
        
        #ADVANCING PHASE BY 90º
        data['jI'][i] = (.0962*data['I1'][i] + .5769*data['I1'][i-2] - .5769*data['I1'][i-4] - .0962*data['I1'][i-6])*(.075*data['period'][i-1] + .54)
        data['jQ'][i] = (.0962*data['Q1'][i] + .5769*data['Q1'][i-2] - .5769*data['Q1'][i-4] - .0962*data['Q1'][i-6])*(.075*data['period'][i-1] + .54)

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
            
        if data['Im'][i] != 0 and data['Re'][i] != 0: data['period'][i] = (2*np.pi)/np.arctan(data['Im'][i]/data['Re'][i])
        if data['period'][i] > 1.5*data['period'][i-1]: data['period'][i] = 1.5*data['period'][i-1]
        if data['period'][i] < .67*data['period'][i-1]: data['period'][i] = .67*data['period'][i-1]
        if data['period'][i] < 6: data['period'][i] = 6
        if data['period'][i] > 50: data['period'][i] = 50

        data['period'][i] = .2*data['period'][i] + .8*data['period'][i-1]
        data['smoothperiod'][i] = .33*data['period'][i] + .67*data['smoothperiod'][i-1]

        #COMPUTE TRENDLINE AS SIMPLE AVERAGE OVER THE DOMINANT CYCLE PERIOD 
        data['DCperiod'] = np.floor(data['smoothperiod'] + .5)
        
        for k in range(i-1):
            data['Itrend'][i] = data['Itrend'][i] + data['price'][i-k]
    
        if data['DCperiod'] > 0: data['Itrend'] = data['Itrend']/data['DCperiod']
        data['trendline'][i] = (4*data['Itrend'][i] + 3*data['Itrend'][i-1] + 2*data['Itrend'][i-2] + data['Itrend'][i-3])/10

        if i < 12: data['trendline'][i] = data['price'][i]

        data['smoothprice'] = 4*data['price'][i] + 3*data['price'][i-1] + 2*data['price'][i-2] + data['price'][i-3]/10
    return data
  """
""" def sar(data):
    data['EPH'] = 0.0
    data['EPL'] = 0.0
    data['USAR'] = data['low'].iloc[0:1]
    data['DSAR'] = data['high'].iloc[0:1]
    data['AFH'] = 0.02
    data['AFL'] = 0.02
    data['PSAR'] = 0.0
    first = True
    for i in range(1,len(data)):
        if first:
            data['EPH'][i] = data['high'].iloc[0:i].max()
            data['EPL'][i] = data['low'].iloc[0:i].min()
            if data['EPH'][i] > data['EPH'][i-1] and data['AFH'][i-1] < 0.19:
                data['AFH'][i] = data['AFH'][i-1] + 0.02
            else: 
                data['AFH'][i] = data['AFH'][i-1]
        
        
            if data['EPL'][i] < data['EPL'][i-1] and data['AFL'][i] < 0.19:
                data['AFL'][i] = data['AFL'][i-1] + 0.02
            else: 
                data['AFL'][i] = data['AFL'][i-1]
            
            data['USAR'][i] = data['USAR'][i-1] + (data['AFH'][i-1] * (data['EPH'][i-1]-data['USAR'][i-1]))
            data['DSAR'][i] = data['DSAR'][i-1] + (data['AFL'][i-1] * (data['EPL'][i-1]-data['DSAR'][i-1]))
            if data['high'][i] > data['DSAR'][i]:
                data['PSAR'][i] = data['USAR'][i]
                data['AFL'][i] = 0.02
                data['AFH'][i] = 0.02
                first = False

            if data['low'][i] < data['USAR'][i]:
                data['PSAR'][i] = data['DSAR'][i]
                data['AFL'][i] = 0.02
                data['AFH'][i] = 0.02
                first = False
            
        else:
            if data['high'][i] > data['DSAR'][i] and data['PSAR'][i] == data['DSAR'][i]:
                data['AFL'][i] = 0.02
                data['AFH'][i] = 0.02

            if data['low'][i] < data['USAR'][i] and data['PSAR'][i] == data['USAR'][i]:
                data['AFL'][i] = 0.02
                data['AFH'][i] = 0.02
            
            if data['EPH'][i] > data['EPH'][i-1] and data['AFH'][i-1] < 0.19:
                data['AFH'][i] = data['AFH'][i-1] + 0.02
            else: 
                data['AFH'][i] = data['AFH'][i-1]
        
            if data['EPL'][i] < data['EPL'][i-1] and data['AFL'][i] < 0.19:
                data['AFL'][i] = data['AFL'][i-1] + 0.02
            else: 
                data['AFL'][i] = data['AFL'][i-1]
            
            data['USAR'][i] = data['USAR'][i-1] + (data['AFH'][i-1] * (data['EPH'][i-1]-data['USAR'][i-1]))
            data['DSAR'][i] = data['DSAR'][i-1] + (data['AFL'][i-1] * (data['EPL'][i-1]-data['DSAR'][i-1]))

            if data['high'][i] > data['DSAR'][i] and data['PSAR'][i] == data['DSAR'][i]:
                data['PSAR'][i] = data['USAR'][i]
                first = False


            if data['low'][i] < data['USAR'][i] and data['PSAR'][i] == data['USAR'][i]:
                data['PSAR'][i] = data['DSAR'][i]
                first = False
    return data
sar(df)
 """

"""
def mama(data, FL = 0.5, SL=0.05):
    data['price'] = (data['high']+data['low'])/2
    #df['F'] = fft( np.array((df['high']+df['low'])/2))
    data['smooth'] = 0.0
    data['detrender'] = 0.0
    data['Q1'] = 0.0
    data['I1'] = 0.0
    data['jI'] = 0.0
    data['jQ'] = 0.0
    data['I'] = 0.0
    data['Q'] = 0.0
    data['I2'] = 0.0
    data['Q2'] = 0.0
    data['1Re'] = 0.0
    data['1Im'] = 0.0
    data['Re'] = 0.0
    data['Im'] = 0.0
    data['Period'] = 0.0
    data['Period2'] = 0.0
    data['SmoothPeriod'] = 0.0
    data['Phase'] = 0.0
    data['DeltaPhase'] = 0.0
    data['alpha'] = 0.0
    data['MAMA'] = 0.0
    data['FAMA'] = 0.0
    for i in range(6, len(data['price'])):
        
        data['smooth'][i] = (4*data['price'][i] + 3*data['price'][i-1] + 2*data['price'][i-2] + data['price'][i-3])/10
        data['detrender'][i] = (0.0962*data['smooth'][i] + 0.5769*data['smooth'][i-1] - 0.5769*data['smooth'][i-3] - 0.962*data['smooth'][i-5])*(0.075*data['Period'][i-1] +0.54)

        #INPHASE AND QUADRATURE COMPONENTS 
        data['Q1'][i] = (0.0962*data['detrender'][i] + 0.5769*data['detrender'][i-2] - 0.5769*data['detrender'][i-4] - 0.0962*data['detrender'][i-6])*(0.075*data['Period'][i-1] + 0.54)
        data['I1'][i] =  data['detrender'][i-3]
        
        #ADVANCING PHASE BY 90º
        data['jI'][i] = (.0962*data['I1'][i] + .5769*data['I1'][i-2] - .5769*data['I1'][i-4] - .0962*data['I1'][i-6])*(.075*data['Period'][i-1] + .54)
        data['jQ'][i] = (.0962*data['Q1'][i] + .5769*data['Q1'][i-2] - .5769*data['Q1'][i-4] - .0962*data['Q1'][i-6])*(.075*data['Period'][i-1] + .54)

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
            
        if data['Im'][i] != 0 and data['Re'][i] != 0: data['Period'][i] = (2*np.pi)/np.arctan(data['Im'][i]/data['Re'][i])
        if data['Period'][i] > 1.5*data['Period'][i-1]: data['Period'][i] = 1.5*data['Period'][i-1]
        if data['Period'][i] < .67*data['Period'][i-1]: data['Period'][i] = .67*data['Period'][i-1]
        if data['Period'][i] < 6: data['Period'][i] = 6
        if data['Period'][i] > 50: data['Period'][i] = 50

        data['Period2'][i] = .2*data['Period'][i] + .8*data['Period'][i-1]
        data['SmoothPeriod'][i] = .33*data['Period2'][i] + .67*data['SmoothPeriod'][i-1]

        if data['I1'][i] != 0: data['Phase'][i] = 180/np.pi*np.arctan(data['Q1'][i] / data['I1'][i])
        data['DeltaPhase'][i] = data['Phase'][i-1] - data['Phase'][i]
        if data['DeltaPhase'][i] < 1: data['DeltaPhase'][i] = 1
        data['alpha'][i] = FL / data['DeltaPhase'][i]
        if data['alpha'][i] < SL: data['alpha'][i] = SL
        
        data['MAMA'][i] = data['alpha'][i]*data['price'][i] + (1 - data['alpha'][i])*data['MAMA'][i-1]
        data['FAMA'][i] = .5*data['alpha'][i]*data['MAMA'][i] + (1 - .5*data['alpha'][i])*data['FAMA'][i-1]
    return data['MAMA'], data['FAMA']
mama(df)
 """

pd.set_option('display.max_rows', None)
print(df)
df.to_csv('indicators.csv', na_rep='Unkown', float_format='%.2f')