import os
import time
from typing import Annotated

import httpx
import logging

import pydantic

import thetadata
from thetadata import ThetaClient
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)

# Get environment variables in .env in project root
load_dotenv()

USERNAME=os.getenv('THETAUSER')
PASSWORD=os.getenv('THETAPASS')

client = ThetaClient(username=USERNAME, passwd=PASSWORD, launch=True)

# exps = client.expirations('AAPL')
# print(exps)

# Indices

# eod = client.indices_historical_eod_report('SPX', 20241201, 20241206)
# print(eod)
#
# snap = client.indices_snapshots_price_snapshot("SPX")
# print(snap)

# trades = client.bulk_snapshots_bulk_greeks(
#     exp=0,
#     root='AAPL',
#
# )
#
# print(trades)


# ohlc_snap = client.indices_snapshots_ohlc_snapshot('SPX')
# print(ohlc_snap)


# stock_hist_quotes = client.stock_historical_quotes(
#     root='AAPL',
#     start_date=20241201,
#     end_date=20241206
# )
# print(stock_hist_quotes['bid_exchange'].unique())
# print(stock_hist_quotes['bid_exchange_name'].unique())
# print(stock_hist_quotes['ask_exchange'].unique())
# print(stock_hist_quotes['ask_exchange_name'].unique())
# print(stock_hist_quotes['ask_condition'].unique())
# print(stock_hist_quotes['ask_condition_name'].unique())
# print(stock_hist_quotes.columns)


# stocks_eod = client.stock_historical_eod_report('AAPL', 20241201, 20241206)


stock_hist_ohlc = client.stock_historical_ohlc("AAPL", 20241201,
                                               20241206)
print(stock_hist_ohlc)


# stock_hist_trades = client.stock_historical_trades("AAPL",
#                                                    start_date=20241202,
#                                                    end_date=20241202)
# print(stock_hist_trades)