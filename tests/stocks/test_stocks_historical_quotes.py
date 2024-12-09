"""Tests for successful stock historical quote requests."""
import pandas as pd
import pytest
from datetime import datetime, date, time
import httpx

from thetadata.exceptions import NoDataError, InvalidParamsError, ServiceError
from thetadata.enums import QuoteCondition, Exchange


def test_stock_historical_quotes_basic(theta_client):
    """Test basic stock quote request with valid parameters."""
    try:
        result = theta_client.stock_historical_quotes(
            root="AAPL",
            start_date=20240101,
            end_date=20240105,
            ivl=900000  # 15-minute intervals
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in [
            'bid', 'ask', 'bid_size', 'ask_size',
            'bid_condition', 'ask_condition',
            'bid_exchange', 'ask_exchange'
        ])
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


@pytest.mark.parametrize("symbol", [
    "AAPL",  # Large cap tech
    "SPY",   # ETF
    "IWM",   # Different ETF
])
def test_stock_historical_quotes_multiple_symbols(theta_client, symbol):
    """Test quote data works for different symbols."""
    try:
        result = theta_client.stock_historical_quotes(
            root=symbol,
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_stock_historical_quotes_date_range(theta_client):
    """Test data retrieval across different date ranges."""
    start_date = 20240101
    end_date = 20240131

    try:
        result = theta_client.stock_historical_quotes(
            root="AAPL",
            start_date=start_date,
            end_date=end_date,
            ivl=900000
        )

        # Check date range
        min_date = result.index.min().date()
        max_date = result.index.max().date()

        assert min_date >= date(2024, 1, 1)
        assert max_date <= date(2024, 1, 31)
        assert result.index.is_monotonic_increasing  # Should be sorted
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_stock_historical_quotes_intervals(theta_client):
    """Test different interval sizes."""
    try:
        # Test 1-minute intervals
        one_min = theta_client.stock_historical_quotes(
            root="AAPL",
            start_date=20240102,
            end_date=20240102,
            ivl=60000
        )

        # Test 5-minute intervals
        five_min = theta_client.stock_historical_quotes(
            root="AAPL",
            start_date=20240102,
            end_date=20240102,
            ivl=300000
        )

        # Five minute intervals should have fewer rows
        assert len(five_min) < len(one_min)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_stock_historical_quotes_rth_filtering(theta_client):
    """Test regular trading hours filtering."""
    try:
        result = theta_client.stock_historical_quotes(
            root="AAPL",
            start_date=20240102,
            end_date=20240102,
            ivl=60000,
            rth=True
        )

        # Check that all times are within RTH (9:30 AM - 4:00 PM ET)
        times = result.index.time
        assert all((time(9, 30) <= t <= time(16, 0)) for t in times)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_stock_historical_quotes_lowercase_symbol(theta_client):
    """Test that lowercase symbols work and match uppercase results."""
    try:
        upper_result = theta_client.stock_historical_quotes(
            root="AAPL",
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        lower_result = theta_client.stock_historical_quotes(
            root="aapl",
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        pd.testing.assert_frame_equal(upper_result, lower_result)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_stock_historical_quotes_whitespace_handling(theta_client):
    """Test that whitespace in symbols is handled correctly."""
    try:
        normal_result = theta_client.stock_historical_quotes(
            root="AAPL",
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        whitespace_result = theta_client.stock_historical_quotes(
            root="  AAPL  ",
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        pd.testing.assert_frame_equal(normal_result, whitespace_result)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")