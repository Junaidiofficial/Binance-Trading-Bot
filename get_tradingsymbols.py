import config #import information from the config file that has the API key from binance
import csv # read files into csv format aka excel 


from binance.client import Client

csvfile = open('symbol_pairs', 'w', newline='')
symbol_writer = csv.writer(csvfile, delimiter=',')



client = Client(config.API_KEY, config.API_SECRET)
exchange_info = client.get_exchange_info()
for s in exchange_info['symbols']:
    print(s['symbol'])
    symbol_writer.writerow(s['symbol'].replace(",", ""))
    


    

csvfile.close()