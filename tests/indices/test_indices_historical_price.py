"""Tests for successful indices historical price requests."""
import pandas as pd
import pytest
from datetime import datetime, date, time
import httpx

from thetadata.exceptions import NoDataError, InvalidParamsError, ServiceError


def test_indices_historical_price_basic(theta_client):
    """Test basic indices price request with valid parameters."""
    try:
        result = theta_client.indices_historical_price(
            root="SPX",
            start_date=20240101,
            end_date=20240105,
            ivl=900000  # 15-minute intervals
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in ['price'])
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_indices_historical_price_data_types(theta_client):
    """Test that returned data has correct types."""
    try:
        result = theta_client.indices_historical_price(
            root="SPX",
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        # Check price column is float
        assert result['price'].dtype == float

        # Check time column
        assert isinstance(result['ms_of_day'].iloc[0], time)

        # Check datetime index
        assert isinstance(result.index[0], datetime)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


@pytest.mark.parametrize("symbol", [
    "SPX",  # S&P 500
    "RUT"  # Russell 2000
])
def test_indices_historical_price_multiple_symbols(theta_client, symbol):
    """Test price data works for different indices."""
    try:
        result = theta_client.indices_historical_price(
            root=symbol,
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_indices_historical_price_date_range(theta_client):
    """Test data retrieval across different date ranges."""
    start_date = 20240101
    end_date = 20240131

    try:
        result = theta_client.indices_historical_price(
            root="SPX",
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


def test_indices_historical_price_intervals(theta_client):
    """Test different interval sizes."""
    try:
        # Test 1-minute intervals
        one_min = theta_client.indices_historical_price(
            root="SPX",
            start_date=20240102,
            end_date=20240102,
            ivl=60000
        )

        # Test 5-minute intervals
        five_min = theta_client.indices_historical_price(
            root="SPX",
            start_date=20240102,
            end_date=20240102,
            ivl=300000
        )

        # Five minute intervals should have fewer rows
        assert len(five_min) < len(one_min)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_indices_historical_price_tick_data(theta_client):
    """Test tick-level data retrieval."""
    try:
        result = theta_client.indices_historical_price(
            root="SPX",
            start_date=20240102,
            end_date=20240102,
            ivl=0  # tick data
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        # RTH should be False for tick data
        assert result.between_time('00:00', '09:30').size > 0 or \
               result.between_time('16:00', '23:59').size > 0
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_indices_historical_price_rth_filtering(theta_client):
    """Test regular trading hours filtering."""
    try:
        result = theta_client.indices_historical_price(
            root="SPX",
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


def test_indices_historical_price_lowercase_symbol(theta_client):
    """Test that lowercase symbols work and match uppercase results."""
    try:
        upper_result = theta_client.indices_historical_price(
            root="SPX",
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        lower_result = theta_client.indices_historical_price(
            root="spx",
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        pd.testing.assert_frame_equal(upper_result, lower_result)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_indices_historical_price_whitespace_handling(theta_client):
    """Test that whitespace in symbols is handled correctly."""
    try:
        normal_result = theta_client.indices_historical_price(
            root="SPX",
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        whitespace_result = theta_client.indices_historical_price(
            root="  SPX  ",
            start_date=20240101,
            end_date=20240105,
            ivl=900000
        )

        pd.testing.assert_frame_equal(normal_result, whitespace_result)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")