import yfinance as yf

date = '2021-01-01'
stock_no = '0050.TW'

stock = yf.Ticker(stock_no)
stock_data = stock.history(start=date)

stock_data.head()

print(stock_data)