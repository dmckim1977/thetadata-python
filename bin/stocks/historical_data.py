import logging
import os

from dotenv import load_dotenv

from thetadata import ThetaClient

# Get environment variables in .env in project root
load_dotenv()


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger('thetadata').setLevel(logging.WARNING)

client = ThetaClient(
    username=os.getenv('THETAUSER'),
    passwd=os.getenv('THETAPASS'))

try:
    # run this right after client is connected
    logger.info('Running standalone client.')
    aapl = client.stock_historical_eod_report(
        20221001, 20241020, 'aapl')

    print(len(aapl), aapl.head())
except Exception as e:
    print(e)
finally:
    client.cleanup()

try:
    # Using context manager
    logger.info('Running context manager.')
    with ThetaClient(
            username=os.getenv('THETAUSER'),
            passwd=os.getenv('THETAPASS')) as client:
        aapl = client.stock_historical_eod_report(20221001, 20241020, 'aapl')
        print(len(aapl), aapl.head())
except Exception as e:
    print(e)
