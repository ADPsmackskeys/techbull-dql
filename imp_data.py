import yfinance as yf
import datetime

data = yf.download ('TCS.NS', start = '2024-07-07', end = datetime.date.today())
data.to_csv ("data/tcs.csv")