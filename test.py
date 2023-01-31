from os import close
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import scipy.stats as sp
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
#Exponential Moving Average
def sma(data, period=20, column='close'):
    data['SMA_' + str(period)] = data[column].rolling(window=period).mean()
    return data[column].rolling(window=period).mean()


def ema(data, period=20, column='close'):
    #formula for ema 
    data['EMA_' + str(period)] = data[column].ewm(span=period, adjust=False).mean()
    return data[column].ewm( span=period, adjust=False).mean()


def prev(data, period=20):
    data['PC'] = data['close'][-1]
    return data
prev(df) 
"""

""" 
def fisher(data, period=9):
    data['price'] = (data['high']+data['low'])/2
    data['H'] = data['price'].rolling(window=period).max()
    data['H'] = data['H'].fillna(0)
    data['L'] = data['price'].rolling(window=period).min()
    data['L'] = data['L'].fillna(0)
    data['X'] = 0.0
    data['FISH'] = 0.0
    data['FISH2'] = 0.0
    def round(val):
        if val > 0.99: val = 0.999
        elif val < -0.99: val = -0.999
        else: val =val
        return val
    for i in range(1, len(data)):
        #data['X'][i] = np.minimum(np.maximum(.66 * ((data['price'][i] - data['L'][i]) / (data['H'][i] - data['L'][i]) - .5) + .67 * data['X'][i-1],-0.999),0.999) note this seems less accurate
        data['X'][i] = round(.66 * ((data['price'][i] - data['L'][i]) / (data['H'][i] - data['L'][i]) - .5) + .67 * data['X'][i-1])
        
        data['FISH'][i] = .5 * np.log((1 + data['X'][i]) / (1 - data['X'][i])) + .5 * data['FISH'][i-1]
        data['FISH2'][i] = data['FISH'][i-1]
    return data
fisher(df)
"""
'''def ItrendV2(data, alpha = .07):
    
    data['price'] = (data['high'] + data['low'])/2
    data['smooth'] = 0.0
    data['ItrendV2'] = 0.0
    data['trigger'] = 0.0

    for i in range(2,7):
        data['ItrendV2'][i] = (data['price'][i] + (2 * data['price'][i-1]) + data['price'][i-2]) / 4 
    for i in range(7,len(data)):
        data['ItrendV2'][i] = ((alpha - (alpha**2 / 4)) * data['price'][i]) + (0.5 * alpha**2 * data['price'][i-1]) - ((alpha - (0.75 * alpha**2)) * data['price'][i-2]) + (2 * (1 - alpha) * data['ItrendV2'][i-1]) - ((1 - alpha)**2 * data['ItrendV2'][i-2])
    
    data['trigger'] =  data['ItrendV2'].rolling(3).apply(lambda x: 2*x[2] - x[0], raw=True)
    
    #data['Itrend'] = data['price'].rolling(3).apply(lambda x:(alpha - (alpha**2)/4)*x[2] + .5*(alpha**2)*x[1] - (.75*(alpha**2))*x[0] + 2*(1 - alpha)*data['Itrend'][1]- ((1 - alpha)**2)*(data['Itrend'][0]), raw=True)
    #data.loc[data['price'] < 7, 'Itrend'] = data['price'].rolling(3).apply(lambda x: ((x[2] + 2*x[1] + x[0])/4), raw=True)
    #data['trigger'] =  data['Itrend'].rolling(3).apply(lambda x: 2*x[2] - x[0], raw=True)
    return data 
ItrendV2(df)'''

""" 
def sineV2(data, alpha=.07):
    data['price'] = (data['high']+data['low'])/2
    data['smooth'] = 0.0
    data['Cycle'] = 0.0
    #data['detrender'] = 0.0
    data['Q1'] = 0.0
    data['I1'] = 0.0
    data['DeltaPhase'] = 0.0
    data['MedianPhase'] = 0.0
    data['DC']= 0.0
    #data['jI'] = 0.0
    #data['jQ'] = 0.0
    #data['I'] = 0.0
    #data['Q'] = 0.0
    data['Period'] = 0.0
    data['Period2'] = 0.0
    data['DCPeriod'] = 0.0
    data['RealPart'] = 0.0
    data['ImagPart'] = 0.0    
    data['DCPhase'] = 0.0
    data['Sine'] = 0.0
    data['Sine_45'] = 0.0
    data['c'] = 0
    data['d'] = 0
    data['C'] = 0.0
    data['D'] = 0.0
    
    data['smooth'] = data['price'].rolling(4).apply(lambda x: (x[3] + 2*x[2] + 2*x[1] + x[0])/6, raw=True)
    #data['smooth'][i] = (data['price'][i] + 2*data['price'][i-1] + 2*data['price'][i-2] + data['price'][i-3])/6
    
    	
    #data['Cycle'] = data['price'].rolling(3).apply(lambda x:(x[2] - 2*x[1] + x[0])/4, raw=True)
    
    #data['Cycle'] = data['smooth'].rolling(3).apply(lambda x:(1-.5*alpha)*(1-.5*alpha)*(x[2]-2*x[1]-x[0])+2*(1-alpha), raw=True)*data['Cycle'].rolling(3).apply(lambda x: x[1]-(1-alpha)*(1-alpha)*x[0], raw=True)
	
    for i in range(2,len(data)):
        if i < 7:
            data['Cycle'][i] = (data['price'][i] - (2*data['price'][i-1]) + data['price'][i-2])/4

        else: data['Cycle'][i] = (((1-(.5*alpha))**2)*(data['smooth'][i]-(2*data['smooth'][i-1])+data['smooth'][i-2]))+(2*(1-alpha)*data['Cycle'][i-1])-(((1-alpha)**2)*data['Cycle'][i-2])
            #data['Cycle'][i] = (1-.5*alpha)*(1-.5*alpha)*(data['smooth'][i]-2*data['smooth'][i-1]-data['smooth'][i-2])+2*(1-alpha)*data['Cycle'][i-1]-(1-alpha)*(1-alpha)*data['Cycle'][i-2]
        
    #data.loc[data['i'].lt(7, axis='index'), 'Cycle'] = data['price'].rolling(3).apply(lambda x:(x[2] - 2*x[1] + x[0])/4, raw=True)
    
    #data.loc[data['i'].gt(6, axis='index'), 'Cycle'] = data['smooth'].rolling(3).apply(lambda x:(1-.5*alpha)*(1-.5*alpha)*(x[2]-2*x[1]-x[0])+2*(1-alpha), raw=True)*data['Cycle'].rolling(3).apply(lambda x: x[1]-(1-alpha)*(1-alpha)*x[0], raw=True)
	
    #data['detrender'][i] = (0.0962*data['smooth'][i] + 0.5769*data['smooth'][i-1] - 0.5769*data['smooth'][i-3] - 0.962*data['smooth'][i-5])*(0.075*data['Period'][i-1] +0.54)
    #INPHASE AND QUADRATURE COMPONENTS 
    #def one(x,y):
        #return ((.0962*x[6]) + (.5769*x[4]) - (.5769*x[2]) - (.0962*x[0])) * (.5 + (.08*y[5]))
    #data['Q1'] = rolling_apply(one, 7, data['Cycle'].values, data['Period'].values)
    #data['Q1'] = data['Cycle'].rolling(7).apply(lambda x: ((.0962*x[6]) + (.5769*x[4]) - (.5769*x[2]) - (.0962*x[0])), raw=True) * data['Period'].rolling(2).apply(lambda y:( .5 + (.08*y[0])), raw=True)
    data['I1'] = data['Cycle'].rolling(4).apply(lambda x: x[0], raw=True)
    #data['I1'][i] = data['Cycle'][i-3]
    
    for i in range(6,len(data)):    
        data['Q1'][i] = ((.0962*data['Cycle'][i]) + (.5769*data['Cycle'][i-2]) - (.5769*data['Cycle'][i-4]) - (.0962*data['Cycle'][i-6]))*(.5 + .08*data['Period'][i-1])
		
        #data['x'][i] = data['Q1'][i]*data['I1']
        if (data['Q1'][i] != 0) and (data['Q1'][i-1] != 0): 
            data['DeltaPhase'][i] =  ((data['I1'][i]/data['Q1'][i]) - (data['I1'][i-1]/data['Q1'][i-1])) / (1 + ((data['I1'][i]*data['I1'][i-1])/(data['Q1'][i]*data['Q1'][i-1])))
        
        else: data['DeltaPhase'][i] = 0
        
        data['DeltaPhase'][i] = np.minimum(np.maximum(data['DeltaPhase'][i],0.1),1.1)

        data['MedianPhase'] = data['DeltaPhase'].rolling(window=5).quantile(0.5)
		
        if data['MedianPhase'][i] == 0: data['DC'][i] = 15 
		
        else: data['DC'][i] = 6.28318 /data['MedianPhase'][i] + .5
		
    #data['Period'] = .33*data['DC']*data['Period'].rolling(2).apply(lambda x: .67*x[0], raw=True)
        data['Period'][i] = .33*data['DC'][i] + .67*data['Period'][i-1] 
	
    #def two(x,y):
        #return .15*x[1] + .85*y[0]
    #data['Period2'] = rolling_apply(two, 2, data['Period'].values, data['Period2'].values)
        data['Period2'][i] = .15*data['Period'][i] + .85*data['Period2'][i-1]
	
    #{Compute Dominant Cycle Phase}
	
    data['DCPeriod'] = np.ceil(data['Period2']).astype(int)
		
    data['c'] = data['c'].index
    data['c'] = data['c'].values[::-1]
    #intec = 0.0
    for i in range(len(data)):
		

        data['c'+str(i)] = data['c'].shift(-((len(data)-1)-i))
        data['c'+str(i)] = data['c'+str(i)].fillna(0)

        for c in range(data['DCPeriod'][i]):

            


            data['RealPart'][i] += np.sin(np.deg2rad(360 *c / data['DCPeriod'][i])) * (data['Cycle'][data['c'+str(i)][c]])
		
            data['ImagPart'][i] += np.cos(np.deg2rad(360 *c / data['DCPeriod'][i])) * (data['Cycle'][data['c'+str(i)][c]])
        
    #data['DCPhase'] = [180/np.pi * np.arctan(data['RealPart'] / data['ImagPart']) if np.abs(x) > 0.001 else 90* np.sign(data['RealPart']) for x in data['ImagPart']]
        
    #data['DCPhase'] = data.ImagPart.map(lambda x: 180/np.pi * np.arctan(data['RealPart'] / data['ImagPart']) if np.abs(x) > 0.001 else 90* np.sign(data['RealPart']))
        if np.abs(data['ImagPart'][i]) > 0.001: data['DCPhase'][i]= 180/np.pi * np.arctan(data['RealPart'][i] / data['ImagPart'][i])
		
        else: data['DCPhase'][i]= 90* np.sign(data['RealPart'][i])
		
    data['DCPhase'] = data['DCPhase'] + 90
	
    for i in range(len(data)):	
    #data['DCPhase'] = [data['DCPhase'] + 180 if x < 0 else data['DCPhase'] for x in data['ImagPart']]
    
        if data['ImagPart'][i] < 0: data['DCPhase'][i] = data['DCPhase'][i] + 180
    
    #data['DCPhase'] = [data['DCPhase'] + 180 if x > 315 else data['DCPhase'] for x in data['DCPhase']]
       
        if data['DCPhase'][i] > 315: data['DCPhase'][i] = data['DCPhase'][i] - 360
		
    data['Sine'] = np.sin(np.deg2rad(data['DCPhase']))
		
    data['Sine_45'] = np.sin(np.deg2rad(data['DCPhase'] + 45))
	
    plt.plot(data['RealPart'], color='red')
	
    plt.plot(data['ImagPart'], color='blue')
	
    plt.show()
	
    return data
sineV2(df)
"""

""" 
def rvi(data, leng = 10):
    data['V1']= 0.0
    data['V2']= 0.0
    data['num'] = 0.0
    data['denom'] = 0.0
    for i in range(3,len(data)):
        data['V1'][i] = ((data['close'][i] - data['open'][i]) + 2*(data['close'][i-1] - data['open'][i-1]) + 2*(data['close'][i-2] - data['open'][i-2]) + (data['close'][i-3] - data['open'][i-3]))/6
        data['V2'][i] = ((data['high'][i] - data['low'][i]) + 2*(data['high'][i-1] - data['low'][i-1]) + 2*(data['high'][i-2] - data['low'][i-2]) + (data['high'][i-3] - data['low'][i-3]))/6
    data['num']+= data['V1'].rolling(leng).mean()
    data['denom'] += data['V2'].rolling(leng).mean()
    data.loc[data['denom'] != 0,  'RVI'] = data['num']/data['denom']
    data['RVItrigger']=0.0
    for z in range(3,len(data)):
        data['RVItrigger'][z] = (data['RVI'][z] + 2*data['RVI'][z-1] + 2*data['RVI'][z-2] + data['RVI'][z-3])/6
    return data
rvi(df) 
"""

""" 
def cg(data, leng = 10):
    data['price'] = (data['high']+data['low'])/2
    data['num'] = 0.0
    data['denom'] = 0.0
    data['c'] = 0
    data['c'] = data['c'].index
    data['c'] = data['c'].values[::-1]
    for i in range(len(data)):
        data['c'+str(i)] = data['c'].shift(-((len(data)-1)-i))
        data['c'+str(i)] = data['c'+str(i)].fillna(0)

        for c in range(leng):
            data['num'][i] += (1+c)*data['price'][data['c'+str(i)][c]]
            data['denom'][i] += data['price'][data['c'+str(i)][c]]

    data.loc[data['denom'] != 0,  'CG'] = (-data['num']/data['denom']) + ((leng+1)/2)
    data['CGtrigger'] = data['CG'].shift(1)
    data['CGsignal'] = 0.0
    for z in range(3,len(data)):
        data['CGsignal'][z] = (data['CG'][z] + 2*data['CG'][z-1] + 2*data['CG'][z-2] + data['CG'][z-3])/6
    return data
cg(df)	
"""

""" 
def cc(data, alpha=0.07, lag=9):
    data['price'] = (data['high']+data['low'])/2
    data['alpha_2'] = 1/(lag+1)
    data['smooth'] = 0.0
    data['Cycle'] = 0.0
    
    data['smooth'] = data['price'].rolling(4).apply(lambda x: (x[3] + 2*x[2] + 2*x[1] + x[0])/6, raw=True)
    for i in range(2,len(data)):
        if i < 7:
            data['Cycle'][i] = (data['price'][i] - (2*data['price'][i-1]) + data['price'][i-2])/4

        else: data['Cycle'][i] = (((1-(.5*alpha))**2)*(data['smooth'][i]-(2*data['smooth'][i-1])+data['smooth'][i-2]))+(2*(1-alpha)*data['Cycle'][i-1])-(((1-alpha)**2)*data['Cycle'][i-2])

    data['CCSignal'] = 0.0
    for i in range(1,len(data)):
        data['CCSignal'][i] = (data['alpha_2'][i]*data['Cycle'][i]) + (1-data['alpha_2'][i])*data['CCSignal'][i-1]
    return data 
cc(df) 
"""
""" 
def stochcc(data, alpha = 0.07, leng = 10):
    data['price'] = (data['high']+data['low'])/2
    data['smooth'] = 0.0
    data['Cycle'] = 0.0
    
    data['smooth'] = data['price'].rolling(4).apply(lambda x: (x[3] + 2*x[2] + 2*x[1] + x[0])/6, raw=True)
    for i in range(2,len(data)):
        if i < 7:
            data['Cycle'][i] = (data['price'][i] - (2*data['price'][i-1]) + data['price'][i-2])/4

        else: data['Cycle'][i] = (((1-(.5*alpha))**2)*(data['smooth'][i]-(2*data['smooth'][i-1])+data['smooth'][i-2]))+(2*(1-alpha)*data['Cycle'][i-1])-(((1-alpha)**2)*data['Cycle'][i-2])
    
    data['MaxCycle'] = data['Cycle'].rolling(leng).max()
    data['MinCycle'] = data['Cycle'].rolling(leng).min()
    
    data.loc[data['MaxCycle'] != data['MinCycle'], 'VCC'] = (data['Cycle'] - data['MinCycle']) / (data['MaxCycle']-data['MinCycle'])
    data['StochCC'] = 0.0
    for i in range(3,len(data)):
        data['StochCC'][i] = ((4*data['VCC'][i]) + (3*data['VCC'][i-1]) + (2*data['VCC'][i-2]) + (data['VCC'][i-3]))/10
    data['StochCC'] = 2*(data['StochCC']-.5)
    data['SCCTrigger'] = 0.0
    for i in range(1,len(data)):
        data['SCCTrigger'][i] = .96*(data['StochCC'][i-1]+.02)
    return data
stochcc(df) 
"""

""" 
def stochcg(data, leng = 10):
    data['price'] = (data['high']+data['low'])/2
    data['num'] = 0.0
    data['denom'] = 0.0

    data['c'] = 0
    data['c'] = data['c'].index
    data['c'] = data['c'].values[::-1]
    for i in range(len(data)):
        data['c'+str(i)] = data['c'].shift(-((len(data)-1)-i))
        data['c'+str(i)] = data['c'+str(i)].fillna(0)

        for c in range(leng):
            data['num'][i] += (1+c)*data['price'][data['c'+str(i)][c]]
            data['denom'][i] += data['price'][data['c'+str(i)][c]]

    data.loc[data['denom'] != 0,  'CG'] = (-data['num']/data['denom']) + ((leng+1)/2)
    
    data['MaxCG'] = data['CG'].rolling(leng).max()
    data['MinCG'] = data['CG'].rolling(leng).min()
    
    data.loc[data['MaxCG'] != data['MinCG'], 'VCG'] = (data['CG'] - data['MinCG']) / (data['MaxCG']-data['MinCG'])
    data['StochCG'] = 0.0
    for i in range(3,len(data)):
        data['StochCG'][i] = ((4*data['VCG'][i]) + (3*data['VCG'][i-1]) + (2*data['VCG'][i-2]) + (data['VCG'][i-3]))/10
    data['StochCG'] = 2*(data['StochCG']-.5)
    data['SCGTrigger'] = 0.0
    for i in range(1,len(data)):
        data['SCGTrigger'][i] = .96*(data['StochCG'][i-1]+.02)
    return data
stochcg(df)	
"""

""" 
def stochrvi(data, leng = 10):
    data['V1']= 0.0
    data['V2']= 0.0
    data['num'] = 0.0
    data['denom'] = 0.0
    for i in range(3,len(data)):
        data['V1'][i] = ((data['close'][i] - data['open'][i]) + 2*(data['close'][i-1] - data['open'][i-1]) + 2*(data['close'][i-2] - data['open'][i-2]) + (data['close'][i-3] - data['open'][i-3]))/6
        data['V2'][i] = ((data['high'][i] - data['low'][i]) + 2*(data['high'][i-1] - data['low'][i-1]) + 2*(data['high'][i-2] - data['low'][i-2]) + (data['high'][i-3] - data['low'][i-3]))/6
    data['num']+= data['V1'].rolling(leng).mean()
    data['denom'] += data['V2'].rolling(leng).mean()
    data.loc[data['denom'] != 0,  'RVI'] = data['num']/data['denom']
    
    data['MaxRVI'] = data['RVI'].rolling(leng).max()
    data['MinRVI'] = data['RVI'].rolling(leng).min()
    
    data.loc[data['MaxRVI'] != data['MinRVI'], 'VRVI'] = (data['RVI'] - data['MinRVI']) / (data['MaxRVI']-data['MinRVI'])
    data['StochRVI'] = 0.0
    for i in range(3,len(data)):
        data['StochRVI'][i] = ((4*data['VRVI'][i]) + (3*data['VRVI'][i-1]) + (2*data['VRVI'][i-2]) + (data['VRVI'][i-3]))/10
    data['StochRVI'] = 2*(data['StochRVI']-.5)
    data['SRVITrigger'] = 0.0
    for i in range(1,len(data)):
        data['SRVITrigger'][i] = .96*(data['StochRVI'][i-1]+.02)
    return data
stochrvi(df)  
"""
""" 
def fishercc(data, alpha = 0.07, leng = 10):
    data['price'] = (data['high']+data['low'])/2
    data['smooth'] = 0.0
    data['Cycle'] = 0.0
    
    data['smooth'] = data['price'].rolling(4).apply(lambda x: (x[3] + 2*x[2] + 2*x[1] + x[0])/6, raw=True)
    for i in range(2,len(data)):
        if i < 7:
            data['Cycle'][i] = (data['price'][i] - (2*data['price'][i-1]) + data['price'][i-2])/4

        else: data['Cycle'][i] = (((1-(.5*alpha))**2)*(data['smooth'][i]-(2*data['smooth'][i-1])+data['smooth'][i-2]))+(2*(1-alpha)*data['Cycle'][i-1])-(((1-alpha)**2)*data['Cycle'][i-2])
    
    data['MaxCycle'] = data['Cycle'].rolling(leng).max()
    data['MinCycle'] = data['Cycle'].rolling(leng).min()
    
    data.loc[data['MaxCycle'] != data['MinCycle'], 'VCC'] = (data['Cycle'] - data['MinCycle']) / (data['MaxCycle']-data['MinCycle'])
    data['StochCC'] = 0.0
    for i in range(3,len(data)):
        data['StochCC'][i] = ((4*data['VCC'][i]) + (3*data['VCC'][i-1]) + (2*data['VCC'][i-2]) + (data['VCC'][i-3]))/10
    
    
    data['FisherCC'] = .5 * np.log((1 + (1.98*(data['StochCC']-.5))) / (1 - (1.98*(data['StochCC']-.5)))) 
    data['FisherCC2'] = data['FisherCC'].shift(1)
    return data
fishercc(df)  
"""

""" 
def fishercg(data, leng = 8):
    data['price'] = (data['high']+data['low'])/2
    data['num'] = 0.0
    data['denom'] = 0.0

    data['c'] = 0
    data['c'] = data['c'].index
    data['c'] = data['c'].values[::-1]
    for i in range(len(data)):
        data['c'+str(i)] = data['c'].shift(-((len(data)-1)-i))
        data['c'+str(i)] = data['c'+str(i)].fillna(0)

        for c in range(leng):
            data['num'][i] += (1+c)*data['price'][data['c'+str(i)][c]]
            data['denom'][i] += data['price'][data['c'+str(i)][c]]

    data.loc[data['denom'] != 0,  'CG'] = (-data['num']/data['denom']) + ((leng+1)/2)
    
    data['MaxCG'] = data['CG'].rolling(leng).max()
    data['MinCG'] = data['CG'].rolling(leng).min()
    
    data.loc[data['MaxCG'] != data['MinCG'], 'VCG'] = (data['CG'] - data['MinCG']) / (data['MaxCG']-data['MinCG'])
    data['VCG'] = data['VCG'].fillna(0)
    data['StochCG'] = 0.0
    for i in range(3,len(data)):
        data['StochCG'][i] = ((4*data['VCG'][i]) + (3*data['VCG'][i-1]) + (2*data['VCG'][i-2]) + (data['VCG'][i-3]))/10
    #data['StochCG'] = 2*(data['StochCG']-.5)

    data['FisherCG'] = .5 * np.log((1 + (1.98*(data['StochCG']-.5))) / (1 - (1.98*(data['StochCG']-.5)))) 
    data['FisherCG2'] = data['FisherCG'].shift(1)

    return data
fishercg(df)	
"""
"""
def fisherrvi(data, leng = 10):
    data['V1']= 0.0
    data['V2']= 0.0
    data['num'] = 0.0
    data['denom'] = 0.0
    for i in range(3,len(data)):
        data['V1'][i] = ((data['close'][i] - data['open'][i]) + 2*(data['close'][i-1] - data['open'][i-1]) + 2*(data['close'][i-2] - data['open'][i-2]) + (data['close'][i-3] - data['open'][i-3]))/6
        data['V2'][i] = ((data['high'][i] - data['low'][i]) + 2*(data['high'][i-1] - data['low'][i-1]) + 2*(data['high'][i-2] - data['low'][i-2]) + (data['high'][i-3] - data['low'][i-3]))/6
    data['num']+= data['V1'].rolling(leng).mean()
    data['denom'] += data['V2'].rolling(leng).mean()
    data.loc[data['denom'] != 0,  'RVI'] = data['num']/data['denom']
    
    data['MaxRVI'] = data['RVI'].rolling(leng).max()
    data['MinRVI'] = data['RVI'].rolling(leng).min()
    
    data.loc[data['MaxRVI'] != data['MinRVI'], 'VRVI'] = (data['RVI'] - data['MinRVI']) / (data['MaxRVI']-data['MinRVI'])
    data['StochRVI'] = 0.0
    for i in range(3,len(data)):
        data['StochRVI'][i] = ((4*data['VRVI'][i]) + (3*data['VRVI'][i-1]) + (2*data['VRVI'][i-2]) + (data['VRVI'][i-3]))/10

    data['FisherRVI'] = .5 * np.log((1 + (1.98*(data['StochRVI']-.5))) / (1 - (1.98*(data['StochRVI']-.5)))) 
    data['FisherRVI2'] = data['FisherRVI'].shift(1)
    return data
fisherrvi(df)
"""
""" 
def acc(data, leng = 10, alpha = 0.07):
    data['price'] = (data['high']+data['low'])/2
    data['smooth'] = 0.0
    data['Cycle'] = 0.0
    data['Q1'] = 0.0
    data['I1'] = 0.0
    data['DeltaPhase'] = 0.0
    data['MedianPhase'] = 0.0
    data['DC']= 0.0
    data['Period'] = 0.0
    data['Period2'] = 0.0


    
    data['smooth'] = data['price'].rolling(4).apply(lambda x: (x[3] + 2*x[2] + 2*x[1] + x[0])/6, raw=True)
    for i in range(2,len(data)):
        if i < 7:
            data['Cycle'][i] = (data['price'][i] - (2*data['price'][i-1]) + data['price'][i-2])/4

        else: data['Cycle'][i] = (((1-(.5*alpha))**2)*(data['smooth'][i]-(2*data['smooth'][i-1])+data['smooth'][i-2]))+(2*(1-alpha)*data['Cycle'][i-1])-(((1-alpha)**2)*data['Cycle'][i-2])
            #data['Cycle'][i] = (1-.5*alpha)*(1-.5*alpha)*(data['smooth'][i]-2*data['smooth'][i-1]-data['smooth'][i-2])+2*(1-alpha)*data['Cycle'][i-1]-(1-alpha)*(1-alpha)*data['Cycle'][i-2]
        
    #data.loc[data['i'].lt(7, axis='index'), 'Cycle'] = data['price'].rolling(3).apply(lambda x:(x[2] - 2*x[1] + x[0])/4, raw=True)
    
    #data.loc[data['i'].gt(6, axis='index'), 'Cycle'] = data['smooth'].rolling(3).apply(lambda x:(1-.5*alpha)*(1-.5*alpha)*(x[2]-2*x[1]-x[0])+2*(1-alpha), raw=True)*data['Cycle'].rolling(3).apply(lambda x: x[1]-(1-alpha)*(1-alpha)*x[0], raw=True)
	
    #data['detrender'][i] = (0.0962*data['smooth'][i] + 0.5769*data['smooth'][i-1] - 0.5769*data['smooth'][i-3] - 0.962*data['smooth'][i-5])*(0.075*data['Period'][i-1] +0.54)
    #INPHASE AND QUADRATURE COMPONENTS 
    #def one(x,y):
        #return ((.0962*x[6]) + (.5769*x[4]) - (.5769*x[2]) - (.0962*x[0])) * (.5 + (.08*y[5]))
    #data['Q1'] = rolling_apply(one, 7, data['Cycle'].values, data['Period'].values)
    #data['Q1'] = data['Cycle'].rolling(7).apply(lambda x: ((.0962*x[6]) + (.5769*x[4]) - (.5769*x[2]) - (.0962*x[0])), raw=True) * data['Period'].rolling(2).apply(lambda y:( .5 + (.08*y[0])), raw=True)
    data['I1'] = data['Cycle'].rolling(4).apply(lambda x: x[0], raw=True)
    #data['I1'][i] = data['Cycle'][i-3]
    
    for i in range(6,len(data)):    
        data['Q1'][i] = ((.0962*data['Cycle'][i]) + (.5769*data['Cycle'][i-2]) - (.5769*data['Cycle'][i-4]) - (.0962*data['Cycle'][i-6]))*(.5 + .08*data['Period'][i-1])
		
        #data['x'][i] = data['Q1'][i]*data['I1']
        if (data['Q1'][i] != 0) and (data['Q1'][i-1] != 0): 
            data['DeltaPhase'][i] =  ((data['I1'][i]/data['Q1'][i]) - (data['I1'][i-1]/data['Q1'][i-1])) / (1 + ((data['I1'][i]*data['I1'][i-1])/(data['Q1'][i]*data['Q1'][i-1])))
        
        else: data['DeltaPhase'][i] = 0
        
        data['DeltaPhase'][i] = np.minimum(np.maximum(data['DeltaPhase'][i],0.1),1.1)

        data['MedianPhase'] = data['DeltaPhase'].rolling(window=5).quantile(0.5)
		
        if data['MedianPhase'][i] == 0: data['DC'][i] = 15 
		
        else: data['DC'][i] = 6.28318 /data['MedianPhase'][i] + .5
		
    #data['Period'] = .33*data['DC']*data['Period'].rolling(2).apply(lambda x: .67*x[0], raw=True)
        data['Period'][i] = .33*data['DC'][i] + .67*data['Period'][i-1] 
	
    #def two(x,y):
        #return .15*x[1] + .85*y[0]
    #data['Period2'] = rolling_apply(two, 2, data['Period'].values, data['Period2'].values)
        data['Period2'][i] = .15*data['Period'][i] + .85*data['Period2'][i-1]
    data['alpha1'] = 2/ (data['Period2']+1)      

    data['ACC'] = 0.0


    for i in range(2,len(data)): 
        if i < 7: 
            data['ACC'][i] = (data['price'][i] - 2*data['price'][i-1] + data['price'][i-2])/4
        else: 
            data['ACC'][i] = ((1-(.5*data['alpha1'][i]))**2 * (data['smooth'][i] - 2*data['smooth'][i-1] + data['smooth'][i-2])) + (2*(1-data['alpha1'][i])*data['ACC'][i-1]) - (((1-data['alpha1'][i])**2)*data['ACC'][i-2])
    data['ACCtrigger'] = data['ACC'].shift(1)
    return data
acc(df)	 
"""
""" 
def acg(data, leng = 10, alpha = 0.07):
    data['price'] = (data['high']+data['low'])/2
    data['smooth'] = 0.0
    data['Cycle'] = 0.0
    data['Q1'] = 0.0
    data['I1'] = 0.0
    data['DeltaPhase'] = 0.0
    data['MedianPhase'] = 0.0
    data['DC']= 0.0
    data['Period'] = 0.0
    data['Period2'] = 0.0
    data['num'] = 0.0
    data['denom'] = 0.0
    data['c'] = 0
    data['c'] = data['c'].index
    data['c'] = data['c'].values[::-1]

    
    data['smooth'] = data['price'].rolling(4).apply(lambda x: (x[3] + 2*x[2] + 2*x[1] + x[0])/6, raw=True)
    for i in range(2,len(data)):
        if i < 7:
            data['Cycle'][i] = (data['price'][i] - (2*data['price'][i-1]) + data['price'][i-2])/4

        else: data['Cycle'][i] = (((1-(.5*alpha))**2)*(data['smooth'][i]-(2*data['smooth'][i-1])+data['smooth'][i-2]))+(2*(1-alpha)*data['Cycle'][i-1])-(((1-alpha)**2)*data['Cycle'][i-2])
            #data['Cycle'][i] = (1-.5*alpha)*(1-.5*alpha)*(data['smooth'][i]-2*data['smooth'][i-1]-data['smooth'][i-2])+2*(1-alpha)*data['Cycle'][i-1]-(1-alpha)*(1-alpha)*data['Cycle'][i-2]
        
    #data.loc[data['i'].lt(7, axis='index'), 'Cycle'] = data['price'].rolling(3).apply(lambda x:(x[2] - 2*x[1] + x[0])/4, raw=True)
    
    #data.loc[data['i'].gt(6, axis='index'), 'Cycle'] = data['smooth'].rolling(3).apply(lambda x:(1-.5*alpha)*(1-.5*alpha)*(x[2]-2*x[1]-x[0])+2*(1-alpha), raw=True)*data['Cycle'].rolling(3).apply(lambda x: x[1]-(1-alpha)*(1-alpha)*x[0], raw=True)
	
    #data['detrender'][i] = (0.0962*data['smooth'][i] + 0.5769*data['smooth'][i-1] - 0.5769*data['smooth'][i-3] - 0.962*data['smooth'][i-5])*(0.075*data['Period'][i-1] +0.54)
    #INPHASE AND QUADRATURE COMPONENTS 
    #def one(x,y):
        #return ((.0962*x[6]) + (.5769*x[4]) - (.5769*x[2]) - (.0962*x[0])) * (.5 + (.08*y[5]))
    #data['Q1'] = rolling_apply(one, 7, data['Cycle'].values, data['Period'].values)
    #data['Q1'] = data['Cycle'].rolling(7).apply(lambda x: ((.0962*x[6]) + (.5769*x[4]) - (.5769*x[2]) - (.0962*x[0])), raw=True) * data['Period'].rolling(2).apply(lambda y:( .5 + (.08*y[0])), raw=True)
    data['I1'] = data['Cycle'].rolling(4).apply(lambda x: x[0], raw=True)
    #data['I1'][i] = data['Cycle'][i-3]
    
    for i in range(6,len(data)):    
        data['Q1'][i] = ((.0962*data['Cycle'][i]) + (.5769*data['Cycle'][i-2]) - (.5769*data['Cycle'][i-4]) - (.0962*data['Cycle'][i-6]))*(.5 + .08*data['Period'][i-1])
		
        #data['x'][i] = data['Q1'][i]*data['I1']
        if (data['Q1'][i] != 0) and (data['Q1'][i-1] != 0): 
            data['DeltaPhase'][i] =  ((data['I1'][i]/data['Q1'][i]) - (data['I1'][i-1]/data['Q1'][i-1])) / (1 + ((data['I1'][i]*data['I1'][i-1])/(data['Q1'][i]*data['Q1'][i-1])))
        
        else: data['DeltaPhase'][i] = 0
        
        data['DeltaPhase'][i] = np.minimum(np.maximum(data['DeltaPhase'][i],0.1),1.1)

        data['MedianPhase'] = data['DeltaPhase'].rolling(window=5).quantile(0.5)
		
        if data['MedianPhase'][i] == 0: data['DC'][i] = 15 
		
        else: data['DC'][i] = 6.28318 /data['MedianPhase'][i] + .5
		
    #data['Period'] = .33*data['DC']*data['Period'].rolling(2).apply(lambda x: .67*x[0], raw=True)
        data['Period'][i] = .33*data['DC'][i] + .67*data['Period'][i-1] 
	
    #def two(x,y):
        #return .15*x[1] + .85*y[0]
    #data['Period2'] = rolling_apply(two, 2, data['Period'].values, data['Period2'].values)
        data['Period2'][i] = .15*data['Period'][i] + .85*data['Period2'][i-1]
    
    data['IntPeriod'] = np.ceil(data['Period2']/2).astype(int)
    for i in range(len(data)):
        data['c'+str(i)] = data['c'].shift(-((len(data)-1)-i))
        data['c'+str(i)] = data['c'+str(i)].fillna(0)

        for c in range(data['IntPeriod'][i]):
            data['num'][i] += (1+c)*data['price'][data['c'+str(i)][c]]
            data['denom'][i] += data['price'][data['c'+str(i)][c]]

    data.loc[data['denom'] != 0,  'ACG'] = (-data['num']/data['denom']) + ((data['IntPeriod']+1)/2)
    data['ACGtrigger'] = data['ACG'].shift(1)
    data['ACGsignal'] = 0.0
    for z in range(3,len(data)):
        data['ACGsignal'][z] = (data['ACG'][z] + 2*data['ACG'][z-1] + 2*data['ACG'][z-2] + data['ACG'][z-3])/6
    return data
acg(df)	 
"""
""" 
def arvi(data, leng = 10, alpha = 0.07):
    data['price'] = (data['high']+data['low'])/2
    data['smooth'] = 0.0
    data['Cycle'] = 0.0
    data['Q1'] = 0.0
    data['I1'] = 0.0
    data['DeltaPhase'] = 0.0
    data['MedianPhase'] = 0.0
    data['DC']= 0.0
    data['Period'] = 0.0
    data['Period2'] = 0.0
    data['IntPeriod'] = 0.0
    data['V1']= 0.0
    data['V2']= 0.0
    data['num'] = 0.0
    data['denom'] = 0.0
    data['c'] = 0

    
    data['smooth'] = data['price'].rolling(4).apply(lambda x: (x[3] + 2*x[2] + 2*x[1] + x[0])/6, raw=True)
    for i in range(2,len(data)):
        if i < 7:
            data['Cycle'][i] = (data['price'][i] - (2*data['price'][i-1]) + data['price'][i-2])/4

        else: data['Cycle'][i] = (((1-(.5*alpha))**2)*(data['smooth'][i]-(2*data['smooth'][i-1])+data['smooth'][i-2]))+(2*(1-alpha)*data['Cycle'][i-1])-(((1-alpha)**2)*data['Cycle'][i-2])
            #data['Cycle'][i] = (1-.5*alpha)*(1-.5*alpha)*(data['smooth'][i]-2*data['smooth'][i-1]-data['smooth'][i-2])+2*(1-alpha)*data['Cycle'][i-1]-(1-alpha)*(1-alpha)*data['Cycle'][i-2]
        
    #data.loc[data['i'].lt(7, axis='index'), 'Cycle'] = data['price'].rolling(3).apply(lambda x:(x[2] - 2*x[1] + x[0])/4, raw=True)
    
    #data.loc[data['i'].gt(6, axis='index'), 'Cycle'] = data['smooth'].rolling(3).apply(lambda x:(1-.5*alpha)*(1-.5*alpha)*(x[2]-2*x[1]-x[0])+2*(1-alpha), raw=True)*data['Cycle'].rolling(3).apply(lambda x: x[1]-(1-alpha)*(1-alpha)*x[0], raw=True)
	
    #data['detrender'][i] = (0.0962*data['smooth'][i] + 0.5769*data['smooth'][i-1] - 0.5769*data['smooth'][i-3] - 0.962*data['smooth'][i-5])*(0.075*data['Period'][i-1] +0.54)
    #INPHASE AND QUADRATURE COMPONENTS 
    #def one(x,y):
        #return ((.0962*x[6]) + (.5769*x[4]) - (.5769*x[2]) - (.0962*x[0])) * (.5 + (.08*y[5]))
    #data['Q1'] = rolling_apply(one, 7, data['Cycle'].values, data['Period'].values)
    #data['Q1'] = data['Cycle'].rolling(7).apply(lambda x: ((.0962*x[6]) + (.5769*x[4]) - (.5769*x[2]) - (.0962*x[0])), raw=True) * data['Period'].rolling(2).apply(lambda y:( .5 + (.08*y[0])), raw=True)
    data['I1'] = data['Cycle'].rolling(4).apply(lambda x: x[0], raw=True)
    #data['I1'][i] = data['Cycle'][i-3]
    
    for i in range(6,len(data)):    
        data['Q1'][i] = ((.0962*data['Cycle'][i]) + (.5769*data['Cycle'][i-2]) - (.5769*data['Cycle'][i-4]) - (.0962*data['Cycle'][i-6]))*(.5 + .08*data['Period'][i-1])
		
        #data['x'][i] = data['Q1'][i]*data['I1']
        if (data['Q1'][i] != 0) and (data['Q1'][i-1] != 0): 
            data['DeltaPhase'][i] =  ((data['I1'][i]/data['Q1'][i]) - (data['I1'][i-1]/data['Q1'][i-1])) / (1 + ((data['I1'][i]*data['I1'][i-1])/(data['Q1'][i]*data['Q1'][i-1])))
        
        else: data['DeltaPhase'][i] = 0
        
        data['DeltaPhase'][i] = np.minimum(np.maximum(data['DeltaPhase'][i],0.1),1.1)

        data['MedianPhase'] = data['DeltaPhase'].rolling(window=5).quantile(0.5)
		
        if data['MedianPhase'][i] == 0: data['DC'][i] = 15 
		
        else: data['DC'][i] = 6.28318 /data['MedianPhase'][i] + .5
		
    #data['Period'] = .33*data['DC']*data['Period'].rolling(2).apply(lambda x: .67*x[0], raw=True)
        data['Period'][i] = .33*data['DC'][i] + .67*data['Period'][i-1] 
	
    #def two(x,y):
        #return .15*x[1] + .85*y[0]
    #data['Period2'] = rolling_apply(two, 2, data['Period'].values, data['Period2'].values)
        data['Period2'][i] = .15*data['Period'][i] + .85*data['Period2'][i-1]
        data['IntPeriod'][i] = np.ceil(((4 * data['Period2'][i]) + (3 * data['Period2'][i-1]) + (2 * data['Period2'][i-3]) + data['Period2'][i-4]) / 20).astype(int)
    
    for i in range(3,len(data)):
        data['V1'][i] = ((data['close'][i] - data['open'][i]) + 2*(data['close'][i-1] - data['open'][i-1]) + 2*(data['close'][i-2] - data['open'][i-2]) + (data['close'][i-3] - data['open'][i-3]))/6
        data['V2'][i] = ((data['high'][i] - data['low'][i]) + 2*(data['high'][i-1] - data['low'][i-1]) + 2*(data['high'][i-2] - data['low'][i-2]) + (data['high'][i-3] - data['low'][i-3]))/6
    
    data['c'] = data['c'].index
    data['c'] = data['c'].values[::-1]
    for i in range(len(data)):
		

        data['c'+str(i)] = data['c'].shift(-((len(data)-1)-i))
        data['c'+str(i)] = data['c'+str(i)].fillna(0)

        for c in range(data['IntPeriod'][i].astype(int)):

            
            data['num'][i] += data['V1'][data['c'+str(i)][c]]
            data['denom'][i] += data['V2'][data['c'+str(i)][c]]
    
    data.loc[data['denom'] != 0,  'ARVI'] = data['num']/data['denom']
    data['Atrigger']=0.0
    for z in range(3,len(data)):
        data['ARVItrigger'][z] = (data['ARVI'][z] + 2*data['ARVI'][z-1] + 2*data['ARVI'][z-2] + data['ARVI'][z-3])/6
    data['ARVIsignal'] = data['ARVI'].shift(1)
    return data
arvi(df)	
"""


#s = (2*np.pi)/(np.arctan(90))
#p = np.log(10)


pd.set_option('display.max_rows', None)
print(df)

df.to_csv('indicators.csv', na_rep='Unkown', float_format='%.2f')