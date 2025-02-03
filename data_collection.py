from binance.client import Client
import pandas as pd
import time


api_key = "---"
api_secret = "xxx"


client = Client(api_key, api_secret)


def fetch_binance_data(symbol, interval, start_date, end_date = None):
    
    try:
        print(f'Fetching data for {symbol}') 
    
        klines = client.get_historical_klines(

            symbol = symbol,
            interval = interval,
            start_str = start_date,
            end_str = end_date
        )
        time.sleep(0.1)

        columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset volume',
                'Taker Buy Quote asset Volume', 'Ignore']
   
        df = pd.DataFrame(klines, columns = columns)

        df['Open Time'] = pd.to_datetime(df['Open Time'], unit = 'ms')
        df['Close Time'] = pd.to_datetime(df['Close Time'], unit = 'ms')
    
        return df

    except Exception as e:
        print(f'Error fetching data: {e}')
        return None






