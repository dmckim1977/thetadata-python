"""Tests for successful expiration requests."""
import pandas as pd
import pytest
from datetime import datetime

def test_expirations_basic(theta_client):
    """Test basic expirations request with valid symbol."""
    result = theta_client.expirations("SPY")

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, int) for x in result)
    assert all(len(str(x)) == 8 for x in result)  # YYYYMMDD format

def test_expirations_lowercase_symbol(theta_client):
    """Test that lowercase symbols work and match uppercase results."""
    upper_result = theta_client.expirations("SPY")
    lower_result = theta_client.expirations("spy")

    assert upper_result == lower_result

@pytest.mark.parametrize("symbol", [
    "AAPL",  # Large cap tech
    "SPY",   # ETF
    "IWM",   # Different ETF
])
def test_expirations_multiple_symbols(theta_client, symbol):
    """Test expirations work for different types of symbols."""
    result = theta_client.expirations(symbol)

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, int) for x in result)
    assert all(20000101 <= x <= 29991231 for x in result)  # Basic date validation
    assert result == sorted(result)  # Should be sorted

def test_expirations_result_format(theta_client):
    """Test detailed aspects of the returned data."""
    result = theta_client.expirations("SPY")

    assert isinstance(result, list)
    assert all(str(x).isdigit() and len(str(x)) == 8 for x in result)
    assert result == sorted(result)  # Should be sorted

    # Test each date is valid
    for date_int in result:
        date_str = str(date_int)
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        # This will raise ValueError if date is invalid
        datetime(year, month, day)

def test_expirations_whitespace_handling(theta_client):
    """Test that whitespace in symbols is handled correctly."""
    normal_result = theta_client.expirations("SPY")
    whitespace_result = theta_client.expirations("  SPY  ")

    assert normal_result == whitespace_result