import datetime
import backtrader as bt
import pandas as pd

class RSIStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.talib.RSI(self.data, period=14)
    def next(self):
        if self.rsi < 30 and not self.position:
            self.buy(size=1)

        if self.rsi > 70 and self.position:
            self.close()




cerebro = bt.Cerebro()


#fromdate = datetime.datetime.strptime('2017-01-01', '%Y-%m-%d')
#todate = datetime.datetime.strptime('2021-06-12', '%Y-%m-%d')
#add (, fromdate=fromdate, todate=todate) into parameters of data if useing above vars.
data = bt.feeds.GenericCSVData(dataname="data/all_time_5min.csv", dtformat=2, compression=5, timeframe=bt.TimeFrame.Minutes)

cerebro.adddata(data)

cerebro.addstrategy(RSIStrategy)
cerebro.run()

cerebro.plot()