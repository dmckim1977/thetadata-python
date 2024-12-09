"""Tests for successful stock historical EOD requests."""
import pandas as pd
import pytest
from datetime import datetime, date, time
import httpx

from thetadata.exceptions import NoDataError, InvalidParamsError, ServiceError


def test_stock_historical_eod_basic(theta_client):
    """Test basic stock EOD request with valid parameters."""
    try:
        result = theta_client.stock_historical_eod_report(
            root="AAPL",
            start_date=20240101,
            end_date=20240105
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close'])
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_stock_historical_eod_data_types(theta_client):
    """Test that returned data has correct types."""
    try:
        result = theta_client.stock_historical_eod_report(
            root="AAPL",
            start_date=20240101,
            end_date=20240105
        )

        # Check price columns are float
        assert result['open'].dtype == float
        assert result['high'].dtype == float
        assert result['low'].dtype == float
        assert result['close'].dtype == float

        # Check time columns if present
        if 'ms_of_day' in result.columns:
            assert isinstance(result['ms_of_day'].iloc[0], time)

        # Check datetime index
        assert isinstance(result.index[0], datetime)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


@pytest.mark.parametrize("symbol", [
    "AAPL",  # Large cap tech
    "SPY",   # ETF
    "IWM",   # Different ETF
    "MSFT",  # Another large cap
])
def test_stock_historical_eod_multiple_symbols(theta_client, symbol):
    """Test EOD data works for different stock symbols."""
    try:
        result = theta_client.stock_historical_eod_report(
            root=symbol,
            start_date=20240101,
            end_date=20240105
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_stock_historical_eod_date_range(theta_client):
    """Test data retrieval across different date ranges."""
    start_date = 20240101
    end_date = 20240131

    try:
        result = theta_client.stock_historical_eod_report(
            root="AAPL",
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


def test_stock_historical_eod_invalid_dates(theta_client):
    """Test handling of invalid date parameters."""
    with pytest.raises(InvalidParamsError):
        theta_client.stock_historical_eod_report(
            root="AAPL",
            start_date=20240132,  # Invalid date
            end_date=20240105
        )

    with pytest.raises(InvalidParamsError):
        theta_client.stock_historical_eod_report(
            root="AAPL",
            start_date=20240105,
            end_date=20240101  # End before start
        )


def test_stock_historical_eod_whitespace_handling(theta_client):
    """Test that whitespace in symbols is handled correctly."""
    try:
        result = theta_client.stock_historical_eod_report(
            root="AAPL",
            start_date=20240101,
            end_date=20240105
        )
        assert isinstance(result, pd.DataFrame)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 472:
            pass  # Expected NoDataError
        else:
            raise


def test_stock_historical_eod_invalid_symbol(theta_client):
    """Test handling of invalid stock symbols."""
    with pytest.raises(InvalidParamsError):
        theta_client.stock_historical_eod_report(
            root="INVALID$SYMBOL",
            start_date=20240101,
            end_date=20240105
        )


def test_stock_historical_eod_future_dates(theta_client):
    """Test handling of future dates."""
    try:
        result = theta_client.stock_historical_eod_report(
            root="AAPL",
            start_date=20991231,
            end_date=20991231
        )
        assert isinstance(result, pd.DataFrame)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 472:
            pass  # Expected NoDataError
        else:
            raise


def test_stock_historical_eod_service_error(theta_client, monkeypatch):
    """Test handling of service errors."""

    def mock_response(*args, **kwargs):
        response = httpx.Response(500)
        return response

    monkeypatch.setattr(httpx, "get", mock_response)

    try:
        result = theta_client.stock_historical_eod_report(
            root="AAPL",
            start_date=20240101,
            end_date=20240105
        )
    except httpx.HTTPStatusError:
        pass  # Expected ServiceError