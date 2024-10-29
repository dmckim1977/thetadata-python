import pytest
import os
import time
from datetime import datetime
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


def test_quote_at_time(client):
    """Test real quote_at_time API calls"""
    quotes = client.quote_at_time(
        start_date=20241021,
        end_date=20241024,
        exp=20241024,
        ivl="15:30:30",
        right='C',
        root="SPXW",
        strike=5850000
    )

    assert quotes is not None
    # Add specific assertions based on the actual data structure returned
    # You might want to check for specific fields or data types


def test_trade_at_time(client):
    """Test real trade_at_time API calls"""
    trades = client.trade_at_time(
        start_date=20241021,
        end_date=20241024,
        exp=20241024,
        ivl="15:30:30",
        right='C',
        root="SPXW",
        strike=5850000
    )

    assert trades is not None
    # Add specific assertions based on the actual data structure returned
    # You might want to check for specific fields or data types


def test_client_status(client):
    """Test client status check"""
    status = client.status()
    print(status)
    assert status is not None
    # Add assertions based on expected status responses


def test_multiple_quote_requests(client):
    """Test multiple quote requests to ensure stability"""
    for _ in range(2):  # Make multiple requests
        quotes = client.quote_at_time(
            start_date=20241021,
            end_date=20241024,
            exp=20241024,
            ivl="15:30:30",
            right='C',
            root="SPXW",
            strike=5850000
        )
        assert quotes is not None
        time.sleep(1)  # Add small delay between requests


def test_multiple_trade_requests(client):
    """Test multiple trade requests to ensure stability"""
    for _ in range(2):  # Make multiple requests
        trades = client.trade_at_time(
            start_date=20241021,
            end_date=20241024,
            exp=20241024,
            ivl="15:30:30",
            right='C',
            root="SPXW",
            strike=5850000
        )
        assert trades is not None
        time.sleep(1)  # Add small delay between requests


def test_invalid_dates():
    """Test handling of invalid date combinations"""
    load_dotenv()
    client = ThetaClient(
        username=os.getenv('THETAUSER'),
        passwd=os.getenv('THETAPASS')
    )
    time.sleep(8)

    with pytest.raises(Exception):  # Replace with specific exception if known
        client.quote_at_time(
            start_date=20241024,  # start_date after end_date
            end_date=20241021,
            exp=20241024,
            ivl="15:30:30",
            right='C',
            root="SPXW",
            strike=5850000
        )

    client.kill()


def test_invalid_tickers():
    """Test handling of invalid date combinations"""
    load_dotenv()
    client = ThetaClient(
        username=os.getenv('THETAUSER'),
        passwd=os.getenv('THETAPASS')
    )
    time.sleep(8)

    with pytest.raises(Exception):  # Replace with specific exception if known
        client.quote_at_time(
            start_date=20241021,  # start_date after end_date
            end_date=20241021,
            exp=20241024,
            ivl="15:30:30",
            right='C',
            root="BADTICKER",
            strike=5850000
        )

    client.kill()