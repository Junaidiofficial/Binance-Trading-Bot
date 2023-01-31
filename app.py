from flask import Flask, render_template, request, flash, redirect, jsonify
import config, csv, datetime
from binance.client import Client
from binance.enums import *

app = Flask(__name__)
app.secret_key = b'bvfsnvnusnvudsnvioimsdivnm234'  # sercet app key 

#set startdate
startdate = "1 Jul, 2020"
#set enddate
enddate = "25 Jun, 2025"

 
client = Client(config.API_KEY, config.API_SECRET) #authentication of the binance account using API Key,  tld='uk'

#Homepage
@app.route("/")
def index():

    return render_template("index.html") # imports information from api to html template
#HISTORY
@app.route('/history')
def history():
    candlesticks = client.get_historical_klines("GBPUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate , enddate )

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#Dashboard
@app.route("/dashboard")
def dashboard():
    title = "Dashboard"
    return render_template("dashboard.html") # account dashboard for client
#APIlogin
@app.route("/APIlogin")
def APIlogin():
    return render_template("APIlogin.html")
#CONTACT
@app.route("/Contact")
def Contact():
    return render_template("contact.html")
#CRYPTOCURRENCY
@app.route("/cryptocurrency")
def Cryptocurrency():
    return render_template("cryptocurrency.html")
#LOGOUTSCREENPAGE
@app.route("/logoutscreenpage")
def logoutscreenpage():
    return render_template("logoutscreenpage.html")





#BUY
@app.route("/buy", methods=['POST']) #when the buy is pressed perform the following function 
def buy():
    print(request.form)
    try:
        order = client.create_order(
            symbol=request.form['symbol'],  # take the selected user symbol to buy 
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=request.form['quantity']) # take the quanitity that the user wants to buy but ensure that its above minimum
    except Exception as e:
        flash(e.message, "error")

    return redirect('/') # send the user back to the index 

#SELL
@app.route("/sell")
def sell():
    print(request.form)
    try:
        order = client.create_order(
            symbol=request.form['symbol'],  # take the selected user symbol to buy 
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=request.form['quantity']) # take the quanitity that the user wants to buy but ensure that its above minimum
    except Exception as e:
        flash(e.message, "error")

    return redirect('/') # send the user back to the index
    
#SETTING
@app.route("/settings")
def setings():

    return "settings" 

#BITCOIN
@app.route('/BTC_USDT')
def BTC():
    title = "Bitcoin"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("bitcoinUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/BTC_USDT')
def historyBTC():
    candlesticks = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate , enddate )

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#ETHEREUM
@app.route('/ETH_USDT')
def ETH():
    title = "Ethereum"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("ethereumUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/ETH_USDT')
def historyETH():
    candlesticks = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#CARDANO
@app.route('/ADA_USDT')
def ADA():
    title = "Cardano"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("cardanoUSDT.html",  title=title, my_balances=balances, symbols=symbols)
@app.route('/history/ADA_USDT')
def historyADA():
    candlesticks = client.get_historical_klines("ADAUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks) 

#DOGECOIN
@app.route('/DOGE_USDT')
def DOGE():
    title = "Dogecoin"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("dogecoinUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/DOGE_USDT')
def historyDOGE():
    candlesticks = client.get_historical_klines("DOGEUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)


#RIPPLE
@app.route('/XRP_USDT')
def RIPPLE():
    title = "Ripple"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("rippleUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/XRP_USDT')
def historyRIPPLE():
    candlesticks = client.get_historical_klines("XRPUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#USD Coin
@app.route('/USD_USDT')
def USD():
    title = "USD Coin"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("usdUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/USD_USDT')
def historyUSD():
    candlesticks = client.get_historical_klines("USDCUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#POLKADOT
@app.route('/DOT_USDT')
def DOT():
    title = "Polkadot"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("polkadotUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/DOT_USDT')
def historyDOT():
    candlesticks = client.get_historical_klines("DOTUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#BINANCE COIN
@app.route('/BNB_USDT')
def BNB():
    title = "Binance Coin"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("binancecoinUSDT.html",  title=title, my_balances=balances, symbols=symbols)
@app.route('/history/BNB_USDT')
def historyBNB():
    candlesticks = client.get_historical_klines("BNBUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#UNISWAP
@app.route('/UNI_USDT')
def UNI():
    title = "Uniswap"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("uniUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/UNI_USDT')
def historyUNI():
    candlesticks = client.get_historical_klines("UNIUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)


#BITCOIN CASH
@app.route('/BCH_USDT')
def BCH():
    title = "Bitcoin Cash"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("bitcoincashUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/BCH_USDT')
def historyBCH():
    candlesticks = client.get_historical_klines("BCHUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#LITECOIN
@app.route('/LTC_USDT')
def LTC():
    title = "Litecoin"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("litecoinUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/LTC_USDT')
def historyLTC():
    candlesticks = client.get_historical_klines("LTCUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#SOLANA
@app.route('/SOL_USDT')
def SOL():
    title = "Solana"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("solanaUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/SOL_USDT')
def historySOL():
    candlesticks = client.get_historical_klines("SOLUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)


#CHAINLINK
@app.route('/LINK_USDT')
def LINK():
    title = "Chainlink"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("chainlinkUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/LINK_USDT')
def historyLINK():
    candlesticks = client.get_historical_klines("LINKUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)


#POLYGON
@app.route('/MATIC_USDT')
def MATIC():
    title = "Polygon"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("polygonUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/MATIC_USDT')
def historyMATIC():
    candlesticks = client.get_historical_klines("MATICUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)




#THETA TOKEN
@app.route('/THETA_USDT')
def THETA():
    title = "Theta Token"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("thetatokenUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/THETA_USDT')
def historyTHETA():
    candlesticks = client.get_historical_klines("THETAUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)


#STELLA 
@app.route('/XLM_USDT')
def XLM():
    title = "Stellar Lumens"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("stellarUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/XLM_USDT')
def historyXLM():
    candlesticks = client.get_historical_klines("XLMUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#VECHAIN
@app.route('/VET_USDT')
def VET():
    title = "Vechain"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("vechainUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/VET_USDT')
def historyVET():
    candlesticks = client.get_historical_klines("VETUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#TRON
@app.route('/TRX_USDT')
def TRX():
    title = "Tron"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("tronUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/TRX_USDT')
def historyTRX():
    candlesticks = client.get_historical_klines("TRXUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#MONERO
@app.route('/XMR_USDT')
def XMRUSDT():
    title = "Monero "
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("moneroUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/XMR_USDT')
def historyXMRUSDT():
    candlesticks = client.get_historical_klines("XMRUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#EOS
@app.route('/EOS_USDT')
def EOSUSDT():
    title = " Eos"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("eosUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/EOS_USDT')
def historyEOSUSDT():
    candlesticks = client.get_historical_klines("EOSUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)


#SHIBA INU
@app.route('/SHIBA_USDT')
def SHIBAUSDT():
    title = " Shiba Inu"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("shibaUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/SHIBA_USDT')
def historySHIBAUSDT():
    candlesticks = client.get_historical_klines("SHIBUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#AAVE
@app.route('/AAVE_USDT')
def AAVEUSDT():
    title = "Aave"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("aaveUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/AAVE_USDT')
def historyAAVEUSDT():
    candlesticks = client.get_historical_klines("AAVEUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#ALGORAND
@app.route('/ALGO_USDT')
def ALGOUSDT():
    title = "Algorand"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("algorandUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/ALGO_USDT')
def historyALGOUSDT():
    candlesticks = client.get_historical_klines("ALGOUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#Terra
@app.route('/TERRA_USDT')
def TERRAUSDT():
    title = "Terra"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("terraUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/TERRA_USDT')
def historyTERRAUSDT():
    candlesticks = client.get_historical_klines("LUNAUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#COSMOS
@app.route('/ATOM_USDT')
def ATOMUSDT():
    title = "Cosmos"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("cosmosUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/ATOM_USDT')
def historyATOMUSDT():
    candlesticks = client.get_historical_klines("ATOMUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#NEO
@app.route('/NEO_USDT')
def NEOUSDT():
    title = "Neo"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("neoUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/NEO_USDT')
def historyNEOUSDT():
    candlesticks = client.get_historical_klines("NEOUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#PANCAKESWAP
@app.route('/CAKE_USDT')
def CAKEUSDT():
    title = "PancakeSwap"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("cakeUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/CAKE_USDT')
def historyCAKEUSDT():
    candlesticks = client.get_historical_klines("CAKEUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#FTX TOKEN
@app.route('/FTT_USDT')
def FTTUSDT():
    title = "Ftx Token"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("fttUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/CAKE_USDT')
def historyFTTUSDT():
    candlesticks = client.get_historical_klines("FTTUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#KLAYTN
@app.route('/KLAY_USDT')
def KLAYUSDT():
    title = "Klaytn"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("klayUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/KLAY_USDT')
def historyKLAYUSDT():
    candlesticks = client.get_historical_klines("KLAYUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#TEZOS
@app.route('/XTZ_USDT')
def TEZOSUSDT():
    title = "Tezos"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("tezosUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/')
def historyTEZOSUSDT():
    candlesticks = client.get_historical_klines("XTZUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#MAKER
@app.route('/MKR_USDT')
def MAKERUSDT():
    title = "Maker"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("makerUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/MKR_USDT')
def historyMAKERUSDT():
    candlesticks = client.get_historical_klines("MKRUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#THETA FUEL
@app.route('/TFUEL_USDT')
def TFUELUSDT():
    title = "Theta Fuel"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("tfuelUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/TFUEL_USDT')
def historyTFUELUSDT():
    candlesticks = client.get_historical_klines("TFUELUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#AVALANCHE
@app.route('/AVAX_USDT')
def AVAXUSDT():
    title = "Avalanche"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("avalancheUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/AVAX_USDT')
def historyAVAXUSDT():
    candlesticks = client.get_historical_klines("AVAXUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#DECRED
@app.route('/DCR_USDT')
def DCRUSDT():
    title = "Decred"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("decredUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/DCR_USDT')
def historyDCRUSDT():
    candlesticks = client.get_historical_klines("DCRUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#WAVES
@app.route('/WAVES_USDT')
def WAVESUSDT():
    title = "Waves"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("wavesUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/WAVES_USDT')
def historyWAVESUSDT():
    candlesticks = client.get_historical_klines("WAVESUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#KUSAMA
@app.route('/KSM_USDT')
def KSMUSDT():
    title = "Kusama"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("kusamaUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/KSM_USDT')
def historyKSMUSDT():
    candlesticks = client.get_historical_klines("KSMUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#THEGRAPH
@app.route('/GRT_USDT')
def GRTUSDT():
    title = "The Graph"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("thegraphUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/GRT_USDT')
def historyGRTUSDT():
    candlesticks = client.get_historical_klines("GRTUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)

#DECRED
@app.route('/FIL_USDT')
def FILUSDT():
    title = "Filecoin"
    acoount = client.get_account()

    balances = acoount['balances']
    
    exchange_info = client.get_exchange_info()
    print(exchange_info)
    symbols = exchange_info['symbols']

    return render_template("filecoinUSDT.html",  title=title, my_balances=balances, symbols=symbols)

@app.route('/history/FIL_USDT')
def historyFILUSDT():
    candlesticks = client.get_historical_klines("FILUSDT", Client.KLINE_INTERVAL_15MINUTE, startdate, enddate)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            "time": data[0] / 1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }

        processed_candlesticks.append(candlestick)

    return jsonify(processed_candlesticks)