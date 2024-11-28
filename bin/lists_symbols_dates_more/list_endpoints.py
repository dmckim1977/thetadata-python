import os
import time

import thetadata
from thetadata import ThetaClient
from dotenv import load_dotenv

# Get environment variables in .env in project root
load_dotenv()

USERNAME=os.getenv('THETAUSER')
PASSWORD=os.getenv('THETAPASS')

client = ThetaClient(username=USERNAME, passwd=PASSWORD)

roots = client.roots(security_type="option")
print(roots)

exps = client.expirations(root="AAPL")
print(exps)

dates = client.option_dates(
    root="AAPL",
    req="quote",
    exp=20241025,
    strike=220000,
    right='C'
    )
print(dates)