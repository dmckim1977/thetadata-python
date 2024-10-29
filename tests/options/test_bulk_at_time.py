import pytest
import os
import time
from thetadata import ThetaClient
from dotenv import load_dotenv


@pytest.fixture(scope="module")
def client():
    """Setup a real client connection that will be used for all tests"""
    load_dotenv()
    username = os.getenv('THETAUSER')
    password = os.getenv('THETAPASS')

    if not username or not password:
        pytest.fail(
            "THETAUSER and THETAPASS environment variables must be set")

    client = ThetaClient(username=username, passwd=password)
    time.sleep(10)  # Wait for connection to establish

    yield client

    # Cleanup
    client.kill()


def test_bulk_quote_at_time(client):
    """Test real quote_at_time API calls"""
    quotes = client.bulk_quote_at_time(
        start_date=20241021,
        end_date=20241024,
        exp=20241024,
        ivl="10:30",
        root="SPXW",
    )

    assert quotes is not None
    # Add specific assertions based on the actual data structure returned
    # You might want to check for specific fields or data types


def test_bulk_trade_at_time(client):
    """Test real trade_at_time API calls"""
    trades = client.bulk_trade_at_time(
        start_date=20241021,
        end_date=20241024,
        exp=20241024,
        ivl="15:30:30",
        root="SPXW",
    )

    assert trades is not None
    # Add specific assertions based on the actual data structure returned
    # You might want to check for specific fields or data types
