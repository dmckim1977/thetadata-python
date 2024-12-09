"""Tests for successful stock historical OHLC requests."""
import pandas as pd
import pytest
from datetime import datetime, date, time
import httpx

from thetadata.exceptions import NoDataError, InvalidParamsError, ServiceError


def test_stock_historical_ohlc_basic(theta_client):
    """Test basic stock OHLC request with valid parameters."""
    try:
        result = theta_client.stock_historical_ohlc(
            root="AAPL",
            start_date=20240101,
            end_date=20240105,
            ivl=900000  # 15-minute intervals
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in
                   ['open', 'high', 'low', 'close', 'volume', 'count'])
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_stock_historical_ohlc_data_types(theta_client):
    """Test that returned data has correct types."""
    try:
        result = theta_client.stock_historical_ohlc(
            root="AAPL",
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        # Check price columns are float
        assert result['open'].dtype == float
        assert result['high'].dtype == float
        assert result['low'].dtype == float
        assert result['close'].dtype == float

        # Check volume and count are int
        assert result['volume'].dtype == int
        assert result['count'].dtype == int

        # Check time column
        assert isinstance(result['ms_of_day'].iloc[0], time)

        # Check datetime index
        assert isinstance(result.index[0], datetime)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


@pytest.mark.parametrize("symbol", [
    "AAPL",  # Large cap tech
    "SPY",  # ETF
    "IWM",  # Different ETF
])
def test_stock_historical_ohlc_multiple_symbols(theta_client, symbol):
    """Test OHLC data works for different symbols."""
    try:
        result = theta_client.stock_historical_ohlc(
            root=symbol,
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_stock_historical_ohlc_date_range(theta_client):
    """Test data retrieval across different date ranges."""
    try:
        result = theta_client.stock_historical_ohlc(
            root="AAPL",
            start_date=20240101,
            end_date=20240131,
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


def test_stock_historical_ohlc_intervals(theta_client):
    """Test different interval sizes."""
    try:
        # Test 1-minute intervals
        one_min = theta_client.stock_historical_ohlc(
            root="AAPL",
            start_date=20240102,
            end_date=20240102,
            ivl=60000
        )

        # Test 5-minute intervals
        five_min = theta_client.stock_historical_ohlc(
            root="AAPL",
            start_date=20240102,
            end_date=20240102,
            ivl=300000
        )

        # Five minute intervals should have fewer rows
        assert len(five_min) < len(one_min)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_stock_historical_ohlc_rth_filtering(theta_client):
    """Test regular trading hours filtering."""
    try:
        result = theta_client.stock_historical_ohlc(
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


def test_stock_historical_ohlc_lowercase_symbol(theta_client):
    """Test that lowercase symbols work and match uppercase results."""
    try:
        upper_result = theta_client.stock_historical_ohlc(
            root="AAPL",
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        lower_result = theta_client.stock_historical_ohlc(
            root="aapl",
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        pd.testing.assert_frame_equal(upper_result, lower_result)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_stock_historical_ohlc_venue_parameter(theta_client):
    """Test venue parameter behavior."""
    try:
        # Test with Nasdaq Basic venue
        nqb_result = theta_client.stock_historical_ohlc(
            root="AAPL",
            start_date=20240101,
            end_date=20240105,
            ivl=900000,
            venue="nqb"
        )

        # Test with UTP/CTA venue
        utp_result = theta_client.stock_historical_ohlc(
            root="AAPL",
            start_date=20240101,
            end_date=20240105,
            ivl=900000,
            venue="utp_cta"
        )

        assert isinstance(nqb_result, pd.DataFrame)
        assert isinstance(utp_result, pd.DataFrame)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")