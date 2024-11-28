import os
import time

from thetadata import ThetaClient
from dotenv import load_dotenv

# Get environment variables in .env in project root
load_dotenv()

USERNAME=os.getenv('THETAUSER')
PASSWORD=os.getenv('THETAPASS')

client = ThetaClient(username=USERNAME, passwd=PASSWORD)

trade_greeks = client.bulk_historical_greeks_bulk_trade_greeks(
        start_date=20241124,
        end_date=20241128,
        exp=20241024,
        root="SPXW",
        ivl=3_600_000,
        perf_boost=True,
    )
print(trade_greeks)