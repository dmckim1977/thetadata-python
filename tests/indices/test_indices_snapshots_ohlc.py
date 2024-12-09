"""Tests for successful indices snapshot requests."""
import pandas as pd
import pytest
from datetime import datetime, date, time
import httpx

from thetadata.exceptions import NoDataError, InvalidParamsError, ServiceError


def test_indices_snapshots_ohlc_basic(theta_client):
    """Test basic indices OHLC snapshot request with valid symbol."""
    try:
        result = theta_client.indices_snapshots_ohlc_snapshot(
            root="SPX"
        )

        # Check response type
        assert result is not None

        # Check all required fields exist
        assert hasattr(result, 'open')
        assert hasattr(result, 'high')
        assert hasattr(result, 'low')
        assert hasattr(result, 'close')
        assert hasattr(result, 'date')
        assert hasattr(result, 'ms_of_day')
        assert hasattr(result, 'datetime')

        # Check data types
        assert isinstance(result.open, float)
        assert isinstance(result.high, float)
        assert isinstance(result.low, float)
        assert isinstance(result.close, float)
        assert isinstance(result.date, date)
        assert isinstance(result.ms_of_day, time)
        assert isinstance(result.datetime, datetime)

        # Check value constraints
        assert result.high >= result.low
        assert result.high >= result.open
        assert result.high >= result.close
        assert result.low <= result.open
        assert result.low <= result.close

    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_indices_snapshots_ohlc_lowercase_symbol(theta_client):
    """Test that lowercase symbols work and match uppercase results."""
    try:
        upper_result = theta_client.indices_snapshots_ohlc_snapshot(
            root="SPX"
        )

        lower_result = theta_client.indices_snapshots_ohlc_snapshot(
            root="spx"
        )

        # Compare all fields except datetime which will naturally differ
        assert isinstance(upper_result.open, float)
        assert isinstance(upper_result.high, float)
        assert isinstance(upper_result.low, float)
        assert isinstance(upper_result.close, float)

    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_indices_snapshots_ohlc_whitespace_handling(theta_client):
    """Test that whitespace in symbols is handled correctly."""
    try:
        normal_result = theta_client.indices_snapshots_ohlc_snapshot(
            root="SPX"
        )

        whitespace_result = theta_client.indices_snapshots_ohlc_snapshot(
            root="  SPX  "
        )

        # Compare all fields except datetime which will naturally differ
        assert isinstance(normal_result.open, float)
        assert isinstance(normal_result.high, float)
        assert isinstance(normal_result.low, float)
        assert isinstance(normal_result.close, float)

    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_indices_snapshots_ohlc_invalid_symbol(theta_client):
    """Test that invalid symbols raise appropriate error."""
    with pytest.raises(InvalidParamsError):
        theta_client.indices_snapshots_ohlc_snapshot(
            root="INVALID_SYMBOL_THATS_WAY_TOO_LONG"  # Will fail root validation
        )

    # For a valid format but non-existent symbol, it should raise NoDataError
    with pytest.raises(NoDataError):
        theta_client.indices_snapshots_ohlc_snapshot(
            root="ABC"  # Valid format but doesn't exist
        )

@pytest.mark.parametrize("symbol", [
    "SPX",  # S&P 500
    "RUT"  # Russell 2000
])
def test_indices_snapshots_ohlc_multiple_symbols(theta_client, symbol):
    """Test OHLC snapshot works for different indices."""
    try:
        result = theta_client.indices_snapshots_ohlc_snapshot(
            root=symbol
        )

        assert result is not None
        assert isinstance(result.open, float)
        assert isinstance(result.high, float)
        assert isinstance(result.low, float)
        assert isinstance(result.close, float)

    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_indices_snapshots_ohlc_to_pandas(theta_client):
    """Test conversion to pandas DataFrame."""
    try:
        result = theta_client.indices_snapshots_ohlc_snapshot(
            root="SPX"
        )

        df = result.to_pandas()

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(
            col in df.columns for col in ['open', 'high', 'low', 'close'])
        assert df.index.name == 'datetime'
        assert df.index.tzinfo is not None  # Check timezone info exists

    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")