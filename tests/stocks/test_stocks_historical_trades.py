"""Tests for successful stock historical trade requests."""
import pandas as pd
import pytest
from datetime import datetime, date, time
import httpx

from thetadata.exceptions import NoDataError, InvalidParamsError, ServiceError
from thetadata.enums import TradeCondition, Exchange


def test_stock_historical_trades_basic(theta_client):
    """Test basic stock trade request with valid parameters."""
    try:
        result = theta_client.stock_historical_trades(
            root="AAPL",
            start_date=20240101,
            end_date=20240105
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in [
            'price', 'size', 'condition', 'exchange', 'sequence'
        ])
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_stock_historical_trades_data_types(theta_client):
    """Test that returned data has correct types."""
    try:
        result = theta_client.stock_historical_trades(
            root="AAPL",
            start_date=20240101,
            end_date=20240105
        )

        assert result['price'].dtype == float
        assert result['size'].dtype == int
        assert isinstance(result['condition'].iloc[0], TradeCondition)
        assert isinstance(result['exchange'].iloc[0], Exchange)
        assert isinstance(result.index[0], datetime)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")


def test_stock_historical_trades_venue_parameter(theta_client):
    """Test venue parameter behavior."""
    try:
        nqb_result = theta_client.stock_historical_trades(
            root="AAPL",
            start_date=20240101,
            end_date=20240105,
            venue="nqb"
        )

        utp_result = theta_client.stock_historical_trades(
            root="AAPL",
            start_date=20240101,
            end_date=20240105,
            venue="utp_cta"
        )

        assert isinstance(nqb_result, pd.DataFrame)
        assert isinstance(utp_result, pd.DataFrame)
    except (httpx.RequestError, httpx.HTTPError) as e:
        pytest.fail(f"Request failed: {str(e)}")