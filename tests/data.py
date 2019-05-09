import pandas as pd
import fix_yahoo_finance as yf

period = 'max'
interval = '1d'
ticker = 'VFSTX'

df = yf.download(ticker, period=period, interval=interval)

print(df.head())
print(df.tail())

df.to_csv('VFSTX')