from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, datetime


# By default the "date" column was in string format,
#  we need to convert it into date-time format
# parse_dates =["date"], converts the "date" column to date-time format

# Resampling works with time-series data only
# so convert "date" column to index
# index_col ="date", makes "date" column
#df = pd.read_csv("data/apple.csv", parse_dates =["date"], index_col ="date")
df = pd.read_csv("data/apple.csv")
#df1m = pd.read_csv("",parse_dates =["date"], index_col ="date") 
#df5m = pd.read_csv("",parse_dates =["date"], index_col ="date")
#df15m = pd.read_csv("",parse_dates =["date"], index_col ="date")
#df30m = pd.read_csv("",parse_dates =["date"], index_col ="date")
#df1h = pd.read_csv("",parse_dates =["date"], index_col ="date")
#df2h = pd.read_csv("",parse_dates =["date"], index_col ="date")
#df3h = pd.read_csv("",parse_dates =["date"], index_col ="date")
#df4h = pd.read_csv("",parse_dates =["date"], index_col ="date")
#df1d = pd.read_csv("",parse_dates =["date"], index_col ="date")
#df1w = pd.read_csv("",parse_dates =["date"], index_col ="date")
#df1M = pd.read_csv("",parse_dates =["date"], index_col ="date")

#Accumulation/Distribution Indicator (A/D)
    #Assess whether a stock is being accumulated or distributed
''' def ad(data):
    #Money Flow Multiplier
    data['MFM'] = ((data['close']-data['low'])-( data['high']-data['close']))/( data['high']-data['low'])
    #Money Flow Volume or Current Money Flow Volume
    data['MFV'] = data['volume'] * data['MFM']
    #Accumulation/Distribution = Previous Acc/Dist + CMFV 
    data['AD'] = data['MFV'].rolling(window=2).sum()
    #return data['AD']
    return data['MFV'].rolling(window=2).sum()


#Exponential Moving Average
def ema(data, period=20, column='close'):
    #formula for ema 
    data['EMA'] = data[column].ewm(span=period, adjust=False).mean()
    return data[column].ewm(span=period, adjust=False).mean()


#TRUE RANGE  
def tr(data):
    data['PC'] = data['close'].shift(periods=1)
    data['HL'] = data['high'] - data['low']
    data['HC'] = np.abs(data['high'] - data['PC'])
    data['LC'] = np.abs(data['low'] - data['PC'])
    data['TR'] = data[['HL', 'HC', 'LC']].max(axis=1)
    return data['TR'] 
tr(df) 

#POSITIVE DIRECTIONAL MOVEMENT 
def pdm(data):
    data['PH'] = data['high'].shift(periods=1)
    data['PDM'] = data['PH'] - data['high']
    return data['PDM']
pdm(df)

#NEGATIVE DIRECTIONAL MOVEMENT 
def ndm(data):
    data['PL'] = data['low'].shift(periods=1)
    data['NDM'] = data['PL'] - data['low']
    return data['NDM']
ndm(df)

#AVERAGE TRUE RANGE (ATR) 
def atr(data, period=15):
    #TRUE RANGE  
    def tr(data):
        data['PC'] = data['close'].shift(periods=1)
        data['HL'] = data['high'] - data['low']
        data['HC'] = np.abs(data['high'] - data['PC'])
        data['LC'] = np.abs(data['low'] - data['PC'])
        data['TR'] = data[['HL', 'HC', 'LC']].max(axis=1)
        return data['TR'] 
    tr(data)
    data['ATR'] = data['TR'].rolling(window=period).mean()
    return data['ATR']
atr(df)

#SMOOTHED POSATIVE DIRECTIONAL MOVEMENT
def spdm(data, period=14):
    data['SPDM'] = data['PDM'].rolling(window=period).sum() - data['PDM'].rolling(window=period).mean() + data['PDM'] 
    return data['SPDM']
spdm(df)

#SMOOTHED NEGATIVE DIRECTIONAL MOVEMENT
def sndm(data, period=14):
    data['SNDM'] = data['NDM'].rolling(window=period).sum() - data['NDM'].rolling(window=period).mean() + data['NDM'] 
    return data['SNDM']

sndm(df)

#POSITIVE DIRECTIONAL INDEX
def pdi(data, period=14):
    data['PDI'] = (data['SPDM']/data['ATR'])*100
    return data['PDI']
pdi(df)

#NEGATIVE DIRECTIONAL INDEX
def ndi(data, period=14):
    data['NDI'] = (data['SNDM']/data['ATR'])*100
    return data['NDI']
ndi(df) 

#OPTIONAL DIRECTIONAL INDEX 
def dx(data, period=14):
    data['DX'] = ((np.abs(data['PDI']-data['NDI'])/np.abs(data['PDI']+data['NDI'])))*100
    return data['DX']
dx(df)
'''
def sma(data, period=20, column='close'):
    data['SMA'] = data[column].rolling(window=period).mean()
    return data[column].rolling(window=period).mean()


def ema(data, period=20, column='close'):
    data['EMA' + str(period)] = data[column].ewm(span=period, adjust=False).mean()
    return data[column].ewm( span=period, adjust=False).mean()

def tema(data, period=20, column='close'):
    data['EMA_1_' + str(period)] = data[column].ewm(span=period, adjust=False).mean()
    data['EMA_2_' + str(period)] = data['EMA_1_' + str(period)].ewm(span=period, adjust=False).mean()
    data['EMA_3_' + str(period)] = data['EMA_2_' + str(period)].ewm(span=period, adjust=False).mean()
    data['TEMA' + str(period)] = (3*data['EMA_1_' + str(period)]) - (3*data['EMA_2_' + str(period)]) + data['EMA_3_' + str(period)]
    return (3*data['EMA_1_' + str(period)]) - (3*data['EMA_2_' + str(period)]) + data['EMA_3_' + str(period)]

def ema2(data, period=20,column='close'):
    data['EMA_1_' + str(period)] = data[column].ewm(span=period, adjust=False).mean()
    data['EMA_2_' + str(period)] = data['EMA_1_' + str(period)].ewm(span=period, adjust=False).mean()
    return data['EMA_1_' + str(period)].ewm(span=period, adjust=False).mean()

def ema3(data, period=20,column='close'):
    data['EMA_1_' + str(period)] = data[column].ewm(span=period, adjust=False).mean()
    data['EMA_2_' + str(period)] = data['EMA_1_' + str(period)].ewm(span=period, adjust=False).mean()
    data['EMA_3_' + str(period)] = data['EMA_2_' + str(period)].ewm(span=period, adjust=False).mean()
    return data['EMA_2_' + str(period)].ewm(span=period, adjust=False).mean()

def tma(data, period=20, column='close'):
    data['SMA_1_' + str(period)] = data[column].rolling(window=period).mean()
    data['TMA_' + str(period)] = data['SMA_1_' + str(period)].rolling(window=period).mean()
    return data['SMA_1_' + str(period)].rolling(window=period).mean()

def wma(df, period=9, column='close' ):

    weights = np.arange(1, period + 1)
    wmas = df[column].rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).to_list()
    df[f'{column}_WMA_{period}'] = wmas
    return df[column].rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).to_list()


#AVERAGE TRUE RANGE (ATR) 
def atr(data, period=14):
    #TRUE RANGE  
    def tr(data):
        data['HL'] = data['high'] - data['low'] 
        data['HC'] = np.abs(data['high'] - data['close'].shift(periods=1))
        data['LC'] = np.abs(data['low'] - data['close'].shift(periods=1))
        data['TR'] = data[['HL', 'HC', 'LC']].max(axis=1)
        data['STR'] = data['TR'].shift(periods=1) - data['TR'].rolling(window=period).mean() + data['TR']
        return data
    tr(data)
    data['ATR'] = round(data['TR'].rolling(window=period).mean(),2)
    return data['ATR'] 
atr(df)

#AVERAGE DIRECTIONAL INDEX 
def adx(data, period=14):
    #UP MOVE
    def um(data):
        df.fillna(0)
        data['PH'] = data['high'].shift(periods=1)
        data['UM'] = data['high'] - data['PH']
        return data
    um(df)

    #DOWN MOVE 
    def dm(data):
        data['PL'] = data['low'].shift(periods=1)
        data['DM'] = data['PL'] - data['low']
    
        return data
    dm(df)


    #PDM
    def pdm(data):
        data.loc[data['UM'] > data['DM'],  'PDM'] = data['UM']
        data.loc[data['UM'] < 0, 'PDM'] = 0
        data.loc[data['UM'] < data['DM'], 'PDM'] = 0
        data['PDM'] = data['PDM'].fillna(0)
        return data 
    pdm(df)

    #NDM
    def ndm(data):
        data.loc[data['DM'] > data['UM'],  'NDM'] = data['DM']
        data.loc[data['DM'] < 0, 'NDM'] = 0
        data.loc[data['DM'] < data['UM'], 'NDM'] = 0
        data['NDM'] = data['NDM'].fillna(0)
        return data 
    ndm(df)


    def ema(data, period=20, column='close', name = ''):
            #formula for ema 
            data['EMA' + str(period) + str(name)] = data[column].ewm(span=period, adjust=False).mean()
            return data[column].ewm(span=period, adjust=False).mean()
    ema(df)

    #AVERAGE TRUE RANGE (ATR) 
    def atr(data, period=14):
        #TRUE RANGE  
        def tr(data):
            data['HL'] = data['high'] - data['low'] 
            data['HC'] = np.abs(data['high'] - data['close'].shift(periods=1))
            data['LC'] = np.abs(data['low'] - data['close'].shift(periods=1))
            data['TR'] = data[['HL', 'HC', 'LC']].max(axis=1)
            data['STR'] = np.nan
            data['STR'] = data['TR'].iloc[0:14].sum()
            data['NTR'] = data['TR'].shift(periods=-1)
            for i in range(len(data)):
                data['STR'][i+1] = data['STR'][i] - data['STR'][i]/14 + data['NTR'][i]
            return data
        tr(data)
        data['ATR'] = round(data['TR'].rolling(window=period).mean(),2)
        return data
    atr(df)

    def satr(data, period=14):
        data['SATR'] = data['ATR'].ewm(span=period, adjust=False).mean()
        return data['SATR']
    satr(df)

    #SMOOTHED POSITIVE DIRECTIONAL MOVEMENT
    def spdm(data, period=14):  
        #data['SPDM'] = data['PDM'].rolling(window=period).sum() - data['PDM'].rolling(window=period).mean() + data['PDM'] 
        data['SPDM'] = np.nan
        data['SPDM'] = data['PDM'].iloc[0:14].sum()
        data['NPDM'] = data['PDM'].shift(periods=-1)
        for i in range(len(data)):
            data['SPDM'][i+1] = data['SPDM'][i] - data['SPDM'][i]/14 + data['NPDM'][i]
        return data['SPDM']
    spdm(df)

    #SMOOTHED NEGATIVE DIRECTIONAL MOVEMENT
    def sndm(data, period=14):
        #data['SNDM'] = data['NDM'].rolling(window=period).sum() - data['NDM'].rolling(window=period).mean() + data['NDM'] 
        data['SNDM'] = np.nan
        data['SNDM'] = data['NDM'].iloc[0:14].sum()
        data['NNDM'] = data['NDM'].shift(periods=-1)
        data['SNDM'].dropna()
        data['NNDM'].dropna()
        for i in range(len(data)):
            data['SNDM'][i+1] = data['SNDM'][i] - data['SNDM'][i]/14 + data['NNDM'][i]
        return data['SNDM'] 
    sndm(df)

    #POSITIVE DIRECTIONAL INDEX
    def pdi(data, period=15):
        data['PDI'] = (data['SPDM']/data['STR'])*100
        #data['PDI'] = (data['PDI'].ewm(span=period, adjust=False).mean())*100
        return data['PDI']
    pdi(df)

    #NEGATIVE DIRECTIONAL INDEX
    def ndi(data, period=15):
        data['NDI'] = (data['SNDM']/data['STR'])*100
        #data['NDI'] = (data['NDI'].ewm(span=period, adjust=False).mean())*100
        return data['NDI']
    ndi(df) 

    #OPTIONAL DIRECTIONAL INDEX 
    def dx(data, period=14):
        data['DX'] = (np.abs(data['PDI']-data['NDI'])/np.abs(data['PDI']+data['NDI']))*100
        data['ADX'] = np.nan
        data['ADX'] = data['DX'].iloc[0:14].sum()
        data['NDX'] = data['DX'].shift(periods=-1)
        for i in range(len(data)):  
            data['ADX'][i+1] =((data['ADX'][i]*13) + data['NDX'][i])/14
            #data['ADX'] = (data['DX'].ewm(span=period, adjust=False).mean())*100
        return data
    dx(df)
adx(df)


#Chiakin Oscillator     
def cho(data):
    #Accumulation/Distribution Indicator (A/D)
    def ad(data):
        #Money Flow Multiplier
        data['MFM'] = ((data['close']-data['low'])-( data['high']-data['close']))/( data['high']-data['low'])
        #Money Flow Volume or Current Money Flow Volume
        data['MFV'] = data['volume'] * data['MFM']
        #Accumulation/Distribution = Previous Acc/Dist + CMFV 
        data['A/D'] =data['MFV'].cumsum()

        return data['A/D']
    ad(df)
    
    def ema(data, period=20, column='close'):
        #formula for ema 
        data['EMA' + str(period)] = data[column].ewm(span=period, adjust=False).mean()
        return data[column].ewm(span=period, adjust=False).mean()

    data['CHO'] = ema(data, 3, 'A/D') - ema(data, 10, 'A/D')
    return data['CHO']
cho(df)






#AVERAGE DIRECTIONAL INDEX RATING
def adxr(data, period=14):
    #AVERAGE DIRECTIONAL INDEX 
    #UP MOVE
    def um(data):
        df.fillna(0)
        data['PH'] = data['high'].shift(periods=1)
        data['UM'] = data['high'] - data['PH']
        return data
    um(df)

    #DOWN MOVE 
    def dm(data):
        data['PL'] = data['low'].shift(periods=1)
        data['DM'] = data['PL'] - data['low']
    
        return data
    dm(df)

    data = data.fillna(0)


    #PDM
    def pdm(data):
        data.loc[data['UM'] > data['DM'],  'PDM'] = data['UM']
        data.loc[data['UM'] < 0, 'PDM'] = 0
        data.loc[data['UM'] < data['DM'], 'PDM'] = 0
        data['PDM'] = data['PDM'].fillna(0)
        return data 
    pdm(df)

    #NDM
    def ndm(data):
        data.loc[data['DM'] > data['UM'],  'NDM'] = data['DM']
        data.loc[data['DM'] < 0, 'NDM'] = 0
        data.loc[data['DM'] < data['UM'], 'NDM'] = 0
        data['NDM'] = data['NDM'].fillna(0)
        return data 
    ndm(df)


    def ema(data, period=20, column='close', name = ''):
            #formula for ema 
            data['EMA' + str(period) + str(name)] = data[column].ewm(span=period, adjust=False).mean()
            return data[column].ewm(span=period, adjust=False).mean()
    ema(df)

    #AVERAGE TRUE RANGE (ATR) 
    def atr(data, period=14):
        #TRUE RANGE  
        def tr(data):
            data['HL'] = data['high'] - data['low'] 
            data['HC'] = np.abs(data['high'] - data['close'].shift(periods=1))
            data['LC'] = np.abs(data['low'] - data['close'].shift(periods=1))
            data['TR'] = data[['HL', 'HC', 'LC']].max(axis=1)
            data['STR'] = np.nan
            data['STR'] = data['TR'].iloc[0:14].sum()
            data['NTR'] = data['TR'].shift(periods=-1)
            for i in range(len(data)):
                data['STR'][i+1] = data['STR'][i] - data['STR'][i]/14 + data['NTR'][i]
            return data
        tr(data)
        data['ATR'] = round(data['TR'].rolling(window=period).mean(),2)
        return data
    atr(df)

    def satr(data, period=14):
        data['SATR'] = data['ATR'].ewm(span=period, adjust=False).mean()
        return data['SATR']
    satr(df)

    #SMOOTHED POSITIVE DIRECTIONAL MOVEMENT
    def spdm(data, period=14):
        #data['SPDM'] = data['PDM'].rolling(window=period).sum() - data['PDM'].rolling(window=period).mean() + data['PDM'] 
        data['SPDM'] = np.nan
        data['SPDM'] = data['PDM'].iloc[0:14].sum()
        data['NPDM'] = data['PDM'].shift(periods=-1)
        for i in range(len(data)):
            data['SPDM'][i+1] = data['SPDM'][i] - data['SPDM'][i]/14 + data['NPDM'][i]
        return data['SPDM']
    spdm(df)

    #SMOOTHED NEGATIVE DIRECTIONAL MOVEMENT
    def sndm(data, period=14):
        #data['SNDM'] = data['NDM'].rolling(window=period).sum() - data['NDM'].rolling(window=period).mean() + data['NDM'] 
        data['SNDM'] = np.nan
        data['SNDM'] = data['NDM'].iloc[0:14].sum()
        data['NNDM'] = data['NDM'].shift(periods=-1)
        data['SNDM'].dropna()
        data['NNDM'].dropna()
        for i in range(len(data)):
            data['SNDM'][i+1] = data['SNDM'][i] - data['SNDM'][i]/14 + data['NNDM'][i]
        return data['SNDM'] 
    sndm(df)

    #POSITIVE DIRECTIONAL INDEX
    def pdi(data, period=15):
        data['PDI'] = (data['SPDM']/data['STR'])*100
        #data['PDI'] = (data['PDI'].ewm(span=period, adjust=False).mean())*100
        return data['PDI']
    pdi(df)

    #NEGATIVE DIRECTIONAL INDEX
    def ndi(data, period=15):
        data['NDI'] = (data['SNDM']/data['STR'])*100
        #data['NDI'] = (data['NDI'].ewm(span=period, adjust=False).mean())*100
        return data['NDI']
    ndi(df) 

    #OPTIONAL DIRECTIONAL INDEX 
    def adx(data, period=14):
        data['DX'] = (np.abs(data['PDI']-data['NDI'])/np.abs(data['PDI']+data['NDI']))*100
        data['ADX'] = np.nan
        data['ADX'] = data['DX'].iloc[0:14].sum()
        data['NDX'] = data['DX'].shift(periods=-1)
        for i in range(len(data)):
            data['ADX'][i+1] =((data['ADX'][i]*13) + data['NDX'][i])/14
        #data['ADX'] = (data['DX'].ewm(span=period, adjust=False).mean())*100
        return data
    adx(df)
    #data['PADX'] = data['ADX'].shift(periods=1)
    #data['ADXR'] = (data['PADX'] + data['ADX'])/2
    return data
adxr(df)

print(df)
def cc(data1, data2, period = 20):
    data = [ data1["close"], data2["close"]]
    headers = ["close1", "close2"]
    data3 = pd.concat(data, axis=1, keys=headers)
    data3['std_close1'] = data3['close1'].rolling(window=period).std()
    data3['std_close2'] = data3['close2'].rolling(window=period).std()
    data3['CC'] = data3['close1'].rolling(window=period).cov(data3['close2'])/((data3['std_close1'])*(data3['std_close2']))
    df['CC'] = data3['CC']
    return df
cc(df,df)


def apo(data, Fperiod=12, Speriod=26):
    data['APO'] = ema(data, Fperiod)-ema(data, Speriod)
    return data
apo(df)

def ppo(data, Fperiod=12, Speriod=26):
    data['PPO'] = (ema(data, Fperiod)-ema(data, Speriod))/ema(data, Speriod)*100
    data['PPO']= data['PPO'].shift(periods=1)
    return data
ppo(df)

def sppo(data, period=9):
    def ppo(data, Fperiod=12, Speriod=26):
        data['PPO'] = (ema(data, Fperiod)-ema(data, Speriod))/ema(data, Speriod)*100
        data['PPO']= data['PPO'].shift(periods=1)
        return data
    ppo(df)
    data['SPPO'] = ema(data, period, column='PPO')
    return data
sppo(df)

def hppo(data):
    data['HPPO'] = data['PPO'] - data['SPPO']
    return data 
hppo(df)

def aroon(data, period= 25):
    data['period_min'] = (data['low'].rolling(window=period,min_periods=0).apply(np.argmin) + data.index - period + 1).astype('Int64')
    data['period_max'] = (data['high'].rolling(window=period,min_periods=0).apply(np.argmax) + data.index - period + 1).astype('Int64')
    data['aroon_smin'] = (data.index - data['period_min']).shift(-period)
    data['aroon_smax'] = (data.index - data['period_max']).shift(-period)
    data['aroon_up'] = ((period - data['aroon_smax'].shift(period))/period)*100
    data['aroon_down'] = ((period - data['aroon_smin'].shift(period))/period)*100
    return data
aroon(df)

def aroono(data, period=25):
    def aroon(data, period= 25):
        data['period_min'] = (data['low'].rolling(window=period,min_periods=0).apply(np.argmin) + data.index - period + 1).astype('Int64')
        data['period_max'] = (data['high'].rolling(window=period,min_periods=0).apply(np.argmax) + data.index - period + 1).astype('Int64')
        data['aroon_smin'] = (data.index - data['period_min']).shift(-period)
        data['aroon_smax'] = (data.index - data['period_max']).shift(-period)
        data['aroon_up'] = ((period - data['aroon_smax'].shift(period))/period)*100
        data['aroon_down'] = ((period - data['aroon_smin'].shift(period))/period)*100
        return data
    aroon(df)
    data['aroono'] = data['aroon_up'] - data['aroon_down']
    return data
aroono(df)

def vwap(data, period=25):
    data['PV'] = ((data['high'] + data['low'] + data['close'])/3) * data['volume']
    data['VWAP'] = data['PV'].cumsum()/ data['volume'].cumsum()
    return data
vwap(df)

def bb(data, period=20, sd=2):
    data['TR'] = (data['high'] + data['low'] + data['close'])/3
    data['MA'] = data['TR'].rolling(window=period).mean()
    data['BBU'] = data['MA'] + sd*data['TR'].rolling(window=period).std()
    data['BBD'] = data['MA'] - sd*data['TR'].rolling(window=period).std()
    return data
bb(df)

def bop(data):
    data['BOP'] = (data['close'] - data['open']) / data['high'] - data['low']

def cci(data, period = 20):
    data['typical_price'] = (data['high'] + data['low'] + data['close'])/3
    #data['typical_price'] =  data['price'].rolling(window = period).sum()
    data['moving_average'] = data['typical_price'].rolling(window=period).mean()
    #data['mean_deviation'] = np.abs(data['typical_price'] - data['moving_average'])
    #data['mean_deviation_'] = data['mean_deviation'].rolling(window=period).mean()
    data['mean_deviation'] = data['typical_price'].rolling(period).apply(lambda x: pd.Series(x).mad())
    data['CCI'] = (data['typical_price'] - data['moving_average'])/(0.015 * data['mean_deviation'])
    return data
cci(df)

def cmo(data, period = 10):
    data['PC'] = data['close'].shift(periods = 1)
    data.loc[(data['close'] - data['PC']) > 0, 'high_close'] = data['close'] - data['PC']
    data.loc[(data['close'] - data['PC']) < 0, 'low_close'] = np.abs(data['close'] - data['PC'])
    data['high_close'] = data['high_close'].fillna(0)
    data['low_close'] = data['low_close'].fillna(0)
    #data.replace(np.nan, 0)
    data['sum_low_close'] =  data['low_close'].rolling(window=period).sum()
    data['sum_high_close'] = data['high_close'].rolling(window=period).sum()
    data['CMO'] = 100*((data['sum_high_close'] - data['sum_low_close'])/(data['sum_high_close'] + data['sum_low_close']))
    return data
cmo(df)

def cc(data1, data2, period = 20):
    df['CC'] = data1['close'].rolling(window=period).corr(data2['close'])
    return df
cc(df,df)

def dema(data, period=9, column = 'close'):
    data['EMA' + str(period)] = data[column].ewm(span=period, adjust=False).mean()
    data['DEMA'] = (2*ema(data, period)) - ema(data, period, column='EMA'+ str(period))
    return data
dema(df)

def lreg(data, period=50):
    X = data.index[len(data)-period:].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = data.iloc[len(data)-period:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    data['Y_pred']=np.zeros(len(data))
    for i in range(period):
        data['Y_pred'][i+(len(data)-period)] = Y_pred[i]
    #plt.scatter(X, Y)
    #plt.plot(X, Y_pred, color='red')
    #plt.show()
    data['Y_pred_up'] = data['Y_pred']+data['close'].iloc[len(data)-period:].std()
    data['Y_pred_down'] = data['Y_pred']-data['close'].iloc[len(data)-period:].std()
    #data['Y_pred_up'] = data['Y_pred'].std()
    return data
lreg(df)

def macd(data, fast=12, slow=26, smoothing=9, ma = ema):
    data['MACD'] = ma(data, fast) - ma(data, slow)
    data['MACD_signal'] = ma(data, smoothing, 'MACD')
    data['MACD_histo'] = data['MACD'] - data['MACD_signal']
    return data
macd(df)

def max(data,period = 20, column = 'close'):
    data['max'] = data[column].rolling(window=period).max()
    return data
max(df)

def min(data,period=20, column='close'):
    data['min'] = data[column].rolling(window=period).min()
    return data
min(df)

def mfi(data, period=14):
    data['TP'] = (data['close'] + data['high'] + data['low'])/3
    data['PTP'] = data['TP'].shift(1)
    data['MF'] = data['TP'] * data['volume']
    data.loc[data['TP'] > data['PTP'],  'PMF'] = data['MF']
    data.loc[data['TP'] < data['PTP'],  'NMF'] = data['MF']
    data['PMF'] = data['PMF'].fillna(0)
    data['NMF'] = data['NMF'].fillna(0)
    data['PMF'+str(period)] = data['PMF'].rolling(window=period).sum()
    data['NMF'+str(period)] = data['NMF'].rolling(window=period).sum()
    data['MFR'] = data['PMF'+str(period)]/data['NMF'+str(period)]
    data['MFI'] = 100 - (100/(1+data['MFR']))
    return data
mfi(df)

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

def mom(data, previous=10):
    data['MOM']=0.0
    for i in range(previous,len(data)):
        data['MOM'][i] = data['close'][i] - data['close'][i-previous]
    return data
mom(df)

def natr(data, period=14):
    #AVERAGE TRUE RANGE (ATR) 
    def atr(data):
        #TRUE RANGE  
        def tr(data):
            data['HL'] = data['high'] - data['low'] 
            data['HC'] = np.abs(data['high'] - data['close'].shift(periods=1))
            data['LC'] = np.abs(data['low'] - data['close'].shift(periods=1))
            data['TR'] = data[['HL', 'HC', 'LC']].max(axis=1)
            data['STR'] = np.nan
            data['STR'] = data['TR'].iloc[0:14].sum()
            return data
        tr(data)
        data['ATR'] = round(data['TR'].rolling(window=period).mean(),2)
        return data
    atr(df)
    data['natr'] = 100* (data['ATR']/data['close'])
    return data
natr(df)

def obv(data):
    OBV = []
    OBV.append(1.65915e+11)

    for i in range(1,len(data)):
        if data.close[i] > data.close[i-1]:
            OBV.append(OBV[-1] + data.volume[i])
        elif data.close[i] < data.close[i-1]:
            OBV.append(OBV[-1] - data.volume[i])
        else: OBV.append(OBV[-1])
    data['OBV'] = OBV
    return data
obv(df)

def roc(data, previous=10,column='close'):
    data['ROC'] = 0.0
    for i in range(previous, len(data)):
        data['ROC'][i] = ((data[column][i] - data[column][i-previous])/data[column][i-previous])*100
    return data
roc(df)

def rocr(data, previous=10):
    data['ROCR'] = 0.0
    for i in range(previous, len(data)):
        data['ROCR'][i] = (data['close'][i]/data['close'][i-previous])
    return data
rocr(df)

def rsi(data, period=14, column='close'):
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
rsi(df)

def trix(data, period1=15, period2=9,column='close', previous=1 ):
    #previous is ment to be equivalent to 1 day so for smaller time periods needs to be changed
    data['EMA_1_' + str(period1)] = data[column].ewm(span=period1, adjust=False).mean()
    data['EMA_2_' + str(period1)] = data['EMA_1_' + str(period1)].ewm(span=period1, adjust=False).mean()
    data['EMA_3_' + str(period1)] = data['EMA_2_' + str(period1)].ewm(span=period1, adjust=False).mean()
    data['TRIX_' + str(period1)] = 0.0
    for i in range(previous,len(data)):
        data['TRIX_' + str(period1)][i] = ((data['EMA_3_' + str(period1)][i]-data['EMA_3_' + str(period1)][i-previous])/data['EMA_3_' + str(period1)][i-previous])*100
    data['TRIX_Signal_' + str(period2)] = data['TRIX_' + str(period1)].ewm(span=period2, adjust=False).mean()
    return data
trix(df)

def will(data, period=14):
    data['%R_'+str(period)] = ((data['high'].rolling(window=period).max() - data['close']) / (data['high'].rolling(window=period).max() - data['low'].rolling(window=period).min())) * -100
    return data
will(df)

def uo(data, period1=7, period2=14, period3=28):
    data['PC'] = data['close'].iloc[0:1]
    for i in range(1,len(data)):
        data['PC'][i] = data['close'][i-1]
    data['BP'] = data['close'] - data[['low','PC']].min(axis=1)
    data['TR'] = data[['high','PC']].max(axis=1) - data[['low','PC']].min(axis=1)
    data['AVG_'+str(period1)] = data['BP'].rolling(window=period1).sum() / data['TR'].rolling(window=period1).sum()
    data['AVG_'+str(period2)] = data['BP'].rolling(window=period2).sum() / data['TR'].rolling(window=period2).sum()
    data['AVG_'+str(period3)] = data['BP'].rolling(window=period3).sum() / data['TR'].rolling(window=period3).sum()
    data['UO'] = 100 * ((4*data['AVG_'+str(period1)])+(2*data['AVG_'+str(period2)])+data['AVG_'+str(period3)]) / (4+2+1)
    return data
uo(df)

def aws(data, period1=5,period2=34,column='HL'):
    data['HL'] = (data['high'] + data['low']) / 2
    data['AO'] = data[column].rolling(window=period1).mean() - data[column].rolling(window=period2).mean()
    return data
aws(df)

def kama(data, period1=10,period2=2,period3=30,column='close'):
    data['PC'] = 0.0
    data['change'] = 0.0
    data['volatility'] = 0.0
    for i in range(period1,len(data)):
        data['PC'][i] = data['close'][i-1]
        data['change'][i] = np.abs(data['close'][i]-data['close'][i-period1])
        data['volatility'][i] = np.abs(data['close'][i]-data['close'][i-1])
    data['volatility'+str(period1)] = data['volatility'].rolling(window=period1).sum()
    data['ER'+str(period1)] = data['change']/data['volatility'+str(period1)]
    data['SC'] = (data['ER'+str(period1)] * ((2/(period2+1))-(2/(period3+1))) + (2/(period3+1)))**2
    data['KAMA'] = data[column].rolling(window=period1).mean()
    for i in range(period1,len(data)):
        data['KAMA'][i] = data['KAMA'][i-1] + data['SC'][i] * (data[column][i] - data['KAMA'][i-1])
    return data 
kama(df) 

def stoch(data, period1=14, period2=2, period3=3):
    data['LL'] = data['low'].rolling(window=period1).min()
    data['HH'] = data['high'].rolling(window=period1).max()
    data['K'] = 100 * ((data['close']-data['LL']) / (data['HH'] - data['LL']))
    data['%K'] = data['K'].rolling(window=period2).mean()
    data['%D'] = data['%K'].rolling(window=period3).mean()
    return data
stoch(df)

def stochf(data, period1=14, period2=1, period3=3):
    data['LL'] = data['low'].rolling(window=period1).min()
    data['HH'] = data['high'].rolling(window=period1).max()
    data['K'] = 100 * ((data['close']-data['LL']) / (data['HH'] - data['LL']))
    data['%K'] = data['K'].rolling(window=period2).mean()
    data['%D'] = data['%K'].rolling(window=period3).mean()
    return data
stoch(df)

def stochs(data, period1=14, period2=3, period3=3):
    data['LL'] = data['low'].rolling(window=period1).min()
    data['HH'] = data['high'].rolling(window=period1).max()
    data['K'] = 100 * ((data['close']-data['LL']) / (data['HH'] - data['LL']))
    data['%K'] = data['K'].rolling(window=period2).mean()
    data['%D'] = data['%K'].rolling(window=period3).mean()
    return data
stoch(df)

def stock_rsi(data, period = 14,period2=3,period3=3):
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
stock_rsi(df)

def wma(df, period=9, column='close' ):

    weights = np.arange(1, period + 1)
    wmas = df[column].rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).to_list()
    df[f'{column}_WMA_{period}'] = wmas
    return df
wma(df)

print(df)

#df['close'].plot()

df.to_csv('indicators.csv', na_rep='Unkown', float_format='%.2f')