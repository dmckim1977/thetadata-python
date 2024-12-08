"""Tests for successful indices historical EOD requests."""
import pandas as pd
import pytest
from datetime import datetime, date, time
import httpx

from thetadata.exceptions import NoDataError, InvalidParamsError, ServiceError

def test_indices_historical_eod_basic(theta_client):
    """Test basic indices EOD request with valid parameters."""
    try:
        result = theta_client.indices_historical_eod_report(
            root="SPX",
            start_date=20240101,
            end_date=20240105
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close'])
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")

def test_indices_historical_eod_data_types(theta_client):
    """Test that returned data has correct types."""
    try:
        result = theta_client.indices_historical_eod_report(
            root="SPX",
            start_date=20240101,
            end_date=20240105
        )

        # Check price columns are float
        assert result['open'].dtype == float
        assert result['high'].dtype == float
        assert result['low'].dtype == float
        assert result['close'].dtype == float

        # Check time columns
        assert isinstance(result['ms_of_day'].iloc[0], time)
        assert isinstance(result['ms_of_day2'].iloc[0], time)

        # If using datetime index
        assert isinstance(result.index[0], datetime)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")

@pytest.mark.parametrize("symbol", [
    "SPX",   # S&P 500
    "RUT"    # Russell 2000
])
def test_indices_historical_eod_multiple_symbols(theta_client, symbol):
    """Test EOD data works for different indices."""
    try:
        result = theta_client.indices_historical_eod_report(
            root=symbol,
            start_date=20240101,
            end_date=20240105
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")

def test_indices_historical_eod_date_range(theta_client):
    """Test data retrieval across different date ranges."""
    start_date = 20240101
    end_date = 20240131

    try:
        result = theta_client.indices_historical_eod_report(
            root="SPX",
            start_date=start_date,
            end_date=end_date
        )

        # Check date range
        min_date = result.index.min().date()
        max_date = result.index.max().date()

        assert min_date >= date(2024, 1, 1)
        assert max_date <= date(2024, 1, 31)
        assert result.index.is_monotonic_increasing  # Should be sorted
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")

def test_indices_historical_eod_lowercase_symbol(theta_client):
    """Test that lowercase symbols work and match uppercase results."""
    try:
        upper_result = theta_client.indices_historical_eod_report(
            root="SPX",
            start_date=20240101,
            end_date=20240105
        )

        lower_result = theta_client.indices_historical_eod_report(
            root="spx",
            start_date=20240101,
            end_date=20240105
        )

        pd.testing.assert_frame_equal(upper_result, lower_result)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")

def test_indices_historical_eod_whitespace_handling(theta_client):
    """Test that whitespace in symbols is handled correctly."""
    try:
        normal_result = theta_client.indices_historical_eod_report(
            root="SPX",
            start_date=20240101,
            end_date=20240105
        )

        whitespace_result = theta_client.indices_historical_eod_report(
            root="  SPX  ",
            start_date=20240101,
            end_date=20240105
        )

        pd.testing.assert_frame_equal(normal_result, whitespace_result)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")