#Python Websocket Client
#   use the data retrive and process data
#   use the TA-LiB to apply RSI indicator to the data 
#       based on  the result received from TA-LIB, if
#                 value > 70 --> binance API reached and Sell 
#                 value < 30 --> binance API reached and buy (oversold)

import websocket, json, pprint, talib, numpy
import config
from binance.client import Client
from binance.enums import *

SOCKET = "wss://stream.binance.com:9443/ws/xrpgbp@kline_1m"   #getting data from binance websocket for 'ethusdt' 1minute data candlestick

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
TRADE_SYMBOL = 'XRPGBP'  #symbol in which the bot will be trading
TRADE_QUANTITY = 24   #quantity  of the trade being brought

        #Stream link in format of '<tradesymbol>@kline_<interval>'
        #intervals for candlestick e.g 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w 
        

closes = [] # stores the appending list of closing values 
in_position = False

client = Client(config.API_KEY, config.API_SECRET)



def order(side, quantity, symbol, order_type=ORDER_TYPE_MARKET) :
    try:
        print('sending order')
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        print(order)
        return True
    except Exception as e:
        return False
    
    return True



def  on_open(ws):  #when a connection to the websocket is made successfully
    print("Open Connection")    #indicates that data can be retrived

def  on_close(ws): #when no connection is made to the websocket
    print("Close Connection")   # indicates no connection can be made and data cannot be retrived 

def on_message(ws, message ): # prints and appends the close value that is received for ea\ch candlestick 
    global closes
    print("Message Received")
    json_message = json.loads(message)
    pprint.pprint(json_message)

    candle = json_message['k']

    is_candle_closed = candle['x']
    close = candle['c']

    if is_candle_closed:
        print("candle closed at {}".format(close))  #waits until the candlestick closes and then prints the value in which the candle stick has closed
        closes.append(float(close)) # saves the close value as a float which can be used by the TA-LIB
        print("closes") #prints close
        print(closes)   #value it closes with 

        if len(closes) > RSI_PERIOD:
            np_closes = numpy.array(closes)
            rsi = talib.RSI(np_closes, RSI_PERIOD)
            print('all RSIs calculated so far')
            print(rsi)
            last_rsi = rsi[-1]
            print('the current rsi is {}'.format(last_rsi))
            
            if last_rsi > RSI_OVERBOUGHT:
                if in_position:
                    print('SELL')
                    #binance sell logic:
                    order_succeeded = order(SIDE_SELL, TRADE_QUANTITY, TRADE_SYMBOL)
                    if order_succeeded:
                        in_position = False
                else:
                    print('we dont own any so nothing to do')
                

            if last_rsi < RSI_OVERSOLD:
                if in_position:
                    print('IT is oversold but u already own it')
                    
                else:
                    print('Oversold going to BUY')
                    #binance buy logic:
                    order_succeeded = order(SIDE_BUY, TRADE_QUANTITY, TRADE_SYMBOL)
                    if order_succeeded: 
                        in_position = True

ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message,) #connect to socket define function
ws.run_forever() #makes the websocket run forever :)