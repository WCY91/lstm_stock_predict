"""
pip install yfinance
"""
import yfinance as yf
df=yf.download('2330.TW',start='2020-01-01',end='2024-01-01')
print(df)
df.to_csv('2330.csv')