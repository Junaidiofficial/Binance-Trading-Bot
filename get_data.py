import config #import information from the config file that has the API key from binance
import csv # read files into csv format aka excel 

from binance.client import Client
client = Client(config.API_KEY, config.API_SECRET) # new instance of client created and import config infomation

#prices = client.get_all_tickers()
#print(prices)

#for price in prices:
    #print(price) 

#set startdate
startdate = "1 Jan, 2019"
#set enddate
enddate = "1 Jan, 2025"


csvfile = open('data/GBPUSDT_15min.csv', 'w', newline='')  # open that 2020_15minutes and write to the file
candlestick_writer = csv.writer(csvfile, delimiter=',')

# get candles information for a particular market and x interval e.g 15mintue
                                                                     # interval is 15mins 
#candlesticks = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("ADAUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("DOGEUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("XRPUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("USDCUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("DOTUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("BNBUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("UNIUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("BCHUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("LTCUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("SOLUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("LINKUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("MATICUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("THETAUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#andlesticks = client.get_historical_klines("XLMUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("VETUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("TRXUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("XMRUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("EOSUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("SHIBUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("AAVEUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("ALGOUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("LUNAUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("ATOMUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("NEOUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("CAKEUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("FTTUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("KLAYUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("XTZUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("MKRUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("TFUELUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("AVAXUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("DCRUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("WAVESUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("KSMUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("GRTUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("TFUELUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
#candlesticks = client.get_historical_klines("FILUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)
candlesticks = client.get_historical_klines("GBPUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate) # currency exchange rate index.html



for candlestick in candlesticks:
    candlestick[0] = candlestick[0] / 1000
    candlestick_writer.writerow(candlestick)

csvfile.close()
