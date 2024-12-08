import os
import time
import httpx

import thetadata
from thetadata import ThetaClient
from dotenv import load_dotenv

# Get environment variables in .env in project root
load_dotenv()

USERNAME=os.getenv('THETAUSER')
PASSWORD=os.getenv('THETAPASS')

client = ThetaClient(username=USERNAME, passwd=PASSWORD)

# exps = client.expirations('AAPL')
# print(exps)

# Indices

eod = client.indices_historical_eod_report('SPX', 20241201, 20241206)
print(eod)

snap = client.indices_snapshots_price_snapshot("SPX")
print(snap)