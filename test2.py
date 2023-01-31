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

'''def optimum_predictor(data, period = 14):
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
    data['detrender2'] = 0.0
    data['alpha'] = 0.0
    data['detrendEMA'] = 0.0 
    data['predict'] = 0.0
    data['smooth2'] = 0.0

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
            
        if data['Im'][i] != 0 and data['Re'][i] != 0: data['period'][i] = 360/np.arctan(data['Im'][i]/data['Re'][i])
        if data['period'][i] > 1.5*data['period'][i-1]: data['period'][i] = 1.5*data['period'][i-1]
        if data['period'][i] < .67*data['period'][i-1]: data['period'][i] = .67*data['period'][i-1]
        if data['period'][i] < 6: data['period'][i] = 6
        if data['period'][i] > 50: data['period'][i] = 50

        data['period'][i] = .2*data['period'][i] + .8*data['period'][i-1]
        data['smoothperiod'][i] = .33*data['period'][i] + .67*data['smoothperiod'][i-1]
    
        #OPTIMUM PREDICTOR 
        data['detrender2'][i] = .5*data['smooth'][i] - .5*data['smooth'][i-2]
        data['smooth2'][i] = (4*data['detrender2'][i] + 3*data['detrender2'][i-1] + 2*data['detrender2'][i-2] + 2*data['detrender2'][i-3])/10

        data['alpha'] = 1 - np.exp(-6.28/data['period'])
        data['detrendEMA'][i] = (data['alpha'][i] * data['smooth2'][i]) + (1 - data['alpha'][i])*data['detrendEMA'][i-1]
        data['predict'] = 1.4*(data['smooth2']-data['detrendEMA'])
    return data
optimum_predictor(df)'''

'''def stoch_rsi(data, period = 14,period2=3,period3=3):
    def rsi(data, period=period, column='close'):
        data['change'] = 0.0
        data['gain'] = 0.0
        data['loss'] = 0.0
        data['AVG_gain'] = 0.0
        data['AVG_loss'] = 0.0
        data['RS'] = 0.0
        data['RSI'] = 0.0
        for i in range(1,len(data)):
            data['change'][i] = data[column][i] - data[column][i-1]
            if data['change'][i] >=0:
                data['gain'][i] = data['change'][i]
            elif data['change'][i] < 0:
                data['loss'][i] = -1*data['change'][i]
            else:
                data['gain'][i] = 0.0
                data['loss'][i] = 0.0
            data['AVG_gain'][i] = data['gain'].iloc[0:period].mean()
            data['AVG_loss'][i] = data['loss'].iloc[0:period].mean()
            data['AVG_gain'][i] = ((data['AVG_gain'][i-1] * (period-1)) + data['gain'][i])/period
            data['AVG_loss'][i] = ((data['AVG_loss'][i-1] * (period-1)) + data['loss'][i])/period
            data['RS'][i] = data['AVG_gain'][i]/ data['AVG_loss'][i]
            data['RSI'][i] = 100 - (100/(1+data['RS'][i]))
        return data
    rsi(data)
    data['stock rsi'] = ((data['RSI']- data['RSI'].rolling(window=period).min())/(data['RSI'].rolling(window=period).max() - data['RSI'].rolling(window=period).min()))*100
    data['%K'] = round(data['stock rsi'].rolling(window=period2).mean(), 6)
    data['%D'] = data['%K'].rolling(window=period3).mean()
    return data
stoch_rsi(df)'''




'''EasyLanguage Code to Plot Instantaneous Trendline

Inputs:	Price((H+L)/2);
		
Vars:	Imult (.635),
		Qmult (.338),
		InPhase(0),
		Quadrature(0),
		Phase(0),
		DeltaPhase(0),
		count(0),
		InstPeriod(0),
		Period(0),
		Trendline(0);

If CurrentBar > 5 then begin

	{Detrend Price}
	Value3 = Price - Price[7];

	{Compute InPhase and Quadrature components}
 	Inphase = 1.25*(Value3[4]  - Imult*Value3[2]) + Imult*InPhase[3];
	Quadrature = Value3[2] - Qmult*Value3 + Qmult*Quadrature[2];

	{Use ArcTangent to compute the current phase}
	If AbsValue(InPhase +InPhase[1]) > 0 then Phase = ArcTangent(AbsValue((Quadrature+Quadrature[1]) / (InPhase+InPhase[1])));

	{Resolve the ArcTangent ambiguity}
	If InPhase < 0 and Quadrature > 0 then Phase = 180 - Phase;
	If InPhase < 0 and Quadrature < 0 then Phase = 180 + Phase;
	If InPhase > 0 and Quadrature < 0 then Phase = 360 - Phase;

	{Compute a differential phase, resolve phase wraparound, and limit delta phase errors}
	DeltaPhase = Phase[1] - Phase;
	If Phase[1] < 90 and Phase > 270 then DeltaPhase = 360 + Phase[1] - Phase;
	If DeltaPhase < 1 then DeltaPhase = 1;
	If DeltaPhase > 60  then Deltaphase = 60;

	{Sum DeltaPhases to reach 360 degrees.  The sum is the instantaneous period.}
	InstPeriod = 0;
	Value4 = 0;
	For count = 0 to 40 begin
		Value4 = Value4 + DeltaPhase[count];
		If Value4 > 360 and InstPeriod = 0 then begin
			InstPeriod = count;
		end;
	end;

	{Resolve Instantaneous Period errors and smooth}
	If InstPeriod = 0 then InstPeriod = InstPeriod[1];
	Value5 = .25*(InstPeriod) + .75*Period[1];

	{Compute Trendline as simple average over the measured dominant cycle period}
	Period = IntPortion(Value5);
	Trendline = 0;
	For  count = 0 to Period - 1 begin
		Trendline = Trendline + Price[count];
	end;
	If Period > 0 then Trendline = Trendline / Period;
	
	Plot1(Trendline, "TR");
end;
'''


'''def wma(Data, period=9):
    weighted = []
    for i in range(len(Data)):
            try:
                total = np.arange(1, period + 1, 1) # weight matrix
                matrix = Data['close'][i - period + 1: i + 1, 3:4]
                matrix = np.ndarray.flatten(matrix)
                matrix = total * matrix # multiplication
                wma = (matrix.sum()) / (total.sum()) # WMA
                weighted = np.append(weighted, wma) # add to array
            except ValueError:
                pass
    total = np.arange(1, period + 1, 1) # weight matrix
    return weighted'''



""" def wma(data, period=9):
    data['price'] = (data['high'] + data['low'])/2
    data['WMA'] = data['price'][0]

    for i in range(1,len(data)):
        data['WMA'][i] += (data['price'][len(data)-1-i])*(len(data)-i)
        data['WMA'][i] = 2*data['WMA'][i]/(i*(i-1)) 
    return data 
wma(df)
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
    #data.loc[data['Im']*data['Re'] != 0,  'period'] = (2*np.pi)/np.arctan(data['Im']/data['Re'])
        #if data['Im'][i] != 0 and data['Re'][i] != 0: data['period'][i] = (2*np.pi)/np.arctan(data['Im'][i]/data['Re'][i])
        #if data['x'][i] != 0: data['period'][i] = (2*np.pi)/np.arctan(data['Im'][i]/data['Re'][i])
        data.loc[~((data['Im'][i] == 0) and (data['Re'][i] == 0)), 'period'] = (2*np.pi)/np.arctan(data['Im'][i]/data['Re'][i])
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

""" def wma(df, period=9, column='close' ):

    weights = np.arange(1, period + 1)
    wmas = df[column].rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).to_list()
    df[f'{column}_WMA_{period}'] = wmas
    return df

wma(df) """


"""
def ItrendV2(data, alpha = .07):
    
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
ItrendV2(df)
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
    data['SNR'] = 0.0

    data['smooth'] = data['price'].rolling(4).apply(lambda x: (4*x[3] + 3*x[2] + 2*x[1] + x[0])/10, raw=True)
    data['smooth'] = data['smooth'].fillna(0)
    #data['smooth'] = (4*data['price'] + 3*data['price'].shift(periods=1) + 2*data['price'].shift(periods=2) + data['price'].shift(periods=3))/10
    data['detrender'] = ((.0962*data['smooth']) + (.5769*data['smooth'].shift(periods=2)) - (.5769*data['smooth'].shift(periods=4)) - (.0962*data['smooth'].shift(periods=6)))*((.075*data['Period'].shift(periods=1)) +0.54)
    #data['detrender'] =  data['smooth'].rolling(7).apply(lambda x: (0.0962*x[6]) + (0.5769*x[4]) - (0.5769*x[2]) - (0.0962*x[0]), raw=True)
    #((0.0962 * smooth) + (0.5769 * nz(smooth[2])) - (0.5769 * nz(smooth[4]))     -  (0.0962 * nz(smooth[6])))  * ((0.075 * nz(period[1])) + 0.54)
        
        #INPHASE AND QUADRATURE COMPONENTS 
    data['Q1'] =  data['detrender'].rolling(7).apply(lambda x: (0.0962*x[6]) + (0.5769*x[4]) - (0.5769*x[2]) - (0.0962*x[0]), raw=True)*((0.075*data['Period'].shift(periods=1)) + 0.54)
                            #((0.0962 * detrender)      + (0.5769 * nz(detrender[2]))     - (0.5769 * nz(detrender[4]))     - (0.0962 * nz(detrender[6])))    * ((0.075 * nz(period[1])) + 0.54)
    data['I1'] =  data['detrender'].shift(periods=3)
    
    #ADVANCING PHASE BY 90º
    data['jI'] = ((0.0962*data['I1']) + (0.5769*data['I1'].shift(periods=2)) - (0.5769*data['I1'].shift(periods=4))- (.0962*data['I1'].shift(periods=6)))*((.075*data['Period'].shift(periods=1)) + .54)
    data['jQ'] = ((0.0962*data['Q1']) + (0.5769*data['Q1'].shift(periods=2)) - (0.5769*data['Q1'].shift(periods=4))- (.0962*data['Q1'].shift(periods=6)))*((.075*data['Period'].shift(periods=1)) + .54)
    #((0.0962 * i1)   + (0.5769 * nz(i1[2]))  - (0.5769 * nz(i1[4]))  - (0.0962 * nz(i1[6]))) * ((0.075 * nz(period[1])) + 0.54)
    
    #PHASOR ADDITION FOR 3 BAR AVERAGING 
    data['I2'] = data['I1'] - data['jQ']
    data['Q2'] = data['Q1'] + data['jI']    
   

    #SMOOTHING I AND Q COMPONENTS 
    data['I2'] = .2*data['I2'] + .8*data['I2'].shift(periods=1)
    data['Q2'] = .2*data['Q2'] + .8*data['Q2'].shift(periods=1)
        
    #HOMODYNE
    data['Re'] = data['I2']*data['I2'].shift(periods=1) + data['Q2']*data['Q2'].shift(periods=1)
    data['Im'] = data['I2']*data['Q2'].shift(periods=1) - data['Q2']*data['I2'].shift(periods=1)
 
    data['Re'] = .2*data['Re'] + .8*data['Re'].shift(periods=1)
    data['Im'] = .2*data['Im'] + .8*data['Im'].shift(periods=1) 
        
    #for i in range(6, len(data)):       
    #period := im != 0 and re != 0 ? 2 * pi / atan(im / re) : 0
    #period := min(max(period, 0.67 * nz(period[1])), 1.5 * nz(period[1]))
    #period := min(max(period, 6), 50)
    #period := (0.2 * period) + (0.8 * nz(period[1])
    
    
    data.loc[np.abs(data['Im']*data['Re']) > 0, 'Period'] = 2 * np.pi / np.arctan(data['Im'] / data['Re'])
    data.loc[np.abs(data['Im']*data['Re']) == 0, 'Period'] = 0
    data['Period'] = np.minimum(np.maximum(data['Period'], 0.67 * data['Period'].shift(periods=1)), 1.5 * data['Period'].shift(periods=1))
    data['Period'] = np.minimum(np.maximum(data['Period'], 6.0), 50.0)
    data['Period'] = (0.2 * data['Period']) + (0.8 * data['Period'].shift(periods=1))

    
    #snr := range > 0 ? (0.25 * ((10 * log(((i1 * i1) + (q1 * q1)) / (range * range)) / log(10)) + 6)) + (0.75 * nz(snr[1])) : 0   

    #sig = src > smooth ? 1 : src < smooth ? -1 : 0
        
        
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


pd.set_option('display.max_rows', None)
print(df)
#df.to_csv('indicators.csv', na_rep='Unkown', float_format='%.2f')