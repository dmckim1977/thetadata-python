"""Tests for successful indices price snapshot requests."""
import pandas as pd
import pytest
from datetime import datetime, date, time
import httpx

from thetadata.exceptions import NoDataError, InvalidParamsError, ServiceError


def test_indices_snapshots_price_basic(theta_client):
    """Test basic indices price snapshot request with valid parameters."""
    try:
        result = theta_client.indices_snapshots_price_snapshot(root="SPX")

        # Check response type
        assert isinstance(result.price, float)
        assert isinstance(result.date, int)
        assert isinstance(result.ms_of_day, int)

    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_indices_snapshots_price_pandas_conversion(theta_client):
    """Test conversion to pandas DataFrame."""
    try:
        result = theta_client.indices_snapshots_price_snapshot(root="SPX")
        df = result.to_pandas()

        # Check DataFrame properties
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == 1  # Should be single row for snapshot
        assert 'price' in df.columns
        assert isinstance(df.index[0], datetime)  # Should have datetime index
        assert df.index[0].tzinfo is not None  # Should be timezone aware

    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


@pytest.mark.parametrize("symbol", [
    "SPX",   # S&P 500
    "RUT"    # Russell 2000
])
def test_indices_snapshots_price_multiple_symbols(theta_client, symbol):
    """Test price snapshot works for different indices."""
    try:
        result = theta_client.indices_snapshots_price_snapshot(root=symbol)
        assert isinstance(result.price, float)
        assert result.price > 0  # Price should be positive

    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_indices_snapshots_price_lowercase_symbol(theta_client):
    """Test that lowercase symbols work and match uppercase results."""
    try:
        upper_result = theta_client.indices_snapshots_price_snapshot(root="SPX")
        lower_result = theta_client.indices_snapshots_price_snapshot(root="spx")

        # Compare key fields
        assert upper_result.date == lower_result.date
        assert abs(upper_result.price - lower_result.price) < 0.01  # Allow small price difference due to time

    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_indices_snapshots_price_whitespace_handling(theta_client):
    """Test that whitespace in symbols is handled correctly."""
    try:
        normal_result = theta_client.indices_snapshots_price_snapshot(root="SPX")
        whitespace_result = theta_client.indices_snapshots_price_snapshot(root="  SPX  ")

        # Compare key fields
        assert normal_result.date == whitespace_result.date
        assert abs(normal_result.price - whitespace_result.price) < 0.01  # Allow small price difference due to time

    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_indices_snapshots_price_invalid_symbol(theta_client):
    """Test that invalid symbols raise appropriate error."""
    with pytest.raises((NoDataError, InvalidParamsError)):
        theta_client.indices_snapshots_price_snapshot(root="INVALID_SYMBOL")