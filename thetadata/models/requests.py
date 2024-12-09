from typing import Optional

from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from .base import DateRange, RootMixin
from ..literals import StockVenue


# region Lists, Symbols, Dates and more

class ExpirationsRequest(RootMixin):
    """Request parameters for getting option expirations.

    :param root: The root/ticker symbol to get expirations for
    """
    pass  # Inherits root validation from RootMixin

# endregion

# region Indices

class IndicesHistoricalEODRequest(DateRange, RootMixin):
    """Request parameters for indices historical EOD data.

    :param start_date: Start date in YYYYMMDD format
    :param end_date: End date in YYYYMMDD format
    :param root: Index symbol/root (e.g., 'SPX', 'NDX')
    """
    pass  # Inherits validation from DateRange and RootMixin


class IndicesHistoricalPriceRequest(DateRange, RootMixin):
    """Request parameters for indices historical price data.

    :param start_date: Start date in YYYYMMDD format
    :param end_date: End date in YYYYMMDD format
    :param root: Index symbol/root (e.g., 'SPX', 'NDX')
    :param ivl: Interval size in milliseconds (e.g., 60000 for 1 minute).
               If 0 or omitted, returns tick-level data.
    :param rth: If True, only return data during regular trading hours (09:30-16:00 ET).
               If ivl is 0, rth is forced to False.
    """
    ivl: int = Field(
        ...,
        ge=0,
        description="Interval size in milliseconds"
    )
    rth: bool = Field(
        default=True,
        description="If True, only return regular trading hours data"
    )

    @field_validator('rth')
    @classmethod
    def validate_rth_with_ivl(cls, v: bool, info: ValidationInfo) -> bool:
        """Validate that rth is False when ivl is 0."""
        ivl = info.data.get('ivl', 0)
        if ivl == 0 and v:
            # Force rth to False for tick-level data
            return False
        return v


class IndicesSnapshotsPriceRequest(RootMixin):
    """Request parameters for index price snapshot.

    :param root: Index symbol/root (e.g., 'SPX', 'NDX')
    """
    pass  # Inherits validation from RootMixin


class IndicesSnapshotsOHLCRequest(RootMixin):
    """Request parameters for index OHLC snapshot.

    :param root: Index symbol/root (e.g., 'SPX', 'NDX')
    """
    pass  # Inherits validation from RootMixin

# endregion

# region Stocks

class StockHistoricalEODRequest(DateRange):
    """Request parameters for stock historical EOD data.

    :param start_date: Start date in YYYYMMDD format
    :param end_date: End date in YYYYMMDD format
    :param root: Stock symbol/root
    """
    root: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock symbol/root"
    )


class StockHistoricalQuoteRequest(DateRange, RootMixin):
    """Request parameters for stock historical quote data.

    :param start_date: Start date in YYYYMMDD format
    :param end_date: End date in YYYYMMDD format
    :param root: Stock symbol/root
    :param ivl: Interval size in milliseconds (e.g., 60000 for 1 minute).
               If 0 or omitted, returns tick-level data.
    :param rth: If True, only return data during regular trading hours (09:30-16:00 ET).
               If ivl is 0, rth is forced to False.
    :param start_time: Optional start time in milliseconds since midnight ET
    :param end_time: Optional end time in milliseconds since midnight ET
    :param venue: Optional venue specification ('nqb' or 'utp_cta')
    """
    ivl: int = Field(
        default=900000,
        ge=0,
        description="Interval size in milliseconds"
    )
    rth: bool = Field(
        default=True,
        description="If True, only return regular trading hours data"
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Start time in milliseconds since midnight ET"
    )
    end_time: Optional[str] = Field(
        default=None,
        description="End time in milliseconds since midnight ET"
    )
    venue: Optional[StockVenue] = Field(
        default=None,
        description="Venue specification (nqb or utp_cta)"
    )

    @field_validator('rth')
    @classmethod
    def validate_rth_with_ivl(cls, v: bool, info: ValidationInfo) -> bool:
        """Validate that rth is False when ivl is 0."""
        ivl = info.data.get('ivl', 0)
        if ivl == 0 and v:
            # Force rth to False for tick-level data
            return False
        return v


class StockHistoricalOHLCRequest(DateRange):
    """Request parameters for stock historical OHLC data.

    :param start_date: Start date in YYYYMMDD format
    :param end_date: End date in YYYYMMDD format
    :param root: Stock symbol/root
    :param ivl: Interval size in milliseconds (e.g., 60000 for 1 minute)
    :param rth: If True, only return regular trading hours data (09:30-16:00 ET)
    :param start_time: Optional start time in milliseconds since midnight ET
    :param end_time: Optional end time in milliseconds since midnight ET
    :param venue: Optional venue specification ('nqb' or 'utp_cta')
    """
    root: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock symbol/root"
    )
    ivl: int = Field(
        ...,
        ge=0,
        description="Interval size in milliseconds"
    )
    rth: bool = Field(
        default=True,
        description="If True, only return regular trading hours data"
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Start time in milliseconds since midnight ET"
    )
    end_time: Optional[str] = Field(
        default=None,
        description="End time in milliseconds since midnight ET"
    )
    venue: Optional[StockVenue] = Field(
        default=None,
        description="Venue specification (nqb or utp_cta)"
    )

    @field_validator('rth')
    @classmethod
    def validate_rth_with_ivl(cls, v: bool, info: ValidationInfo) -> bool:
        """Validate that rth is False when ivl is 0."""
        ivl = info.data.get('ivl', 0)
        if ivl == 0 and v:
            return False
        return v


class StockHistoricalTradeRequest(DateRange, RootMixin):
    """Request parameters for stock historical trade data.

    :param start_date: Start date in YYYYMMDD format
    :param end_date: End date in YYYYMMDD format
    :param root: Stock symbol/root
    :param start_time: Optional start time in milliseconds since midnight ET
    :param end_time: Optional end time in milliseconds since midnight ET
    :param venue: Optional venue specification ('nqb' or 'utp_cta')
    """
    start_time: Optional[str] = Field(
        default=None,
        description="Start time in milliseconds since midnight ET"
    )
    end_time: Optional[str] = Field(
        default=None,
        description="End time in milliseconds since midnight ET"
    )
    venue: Optional[StockVenue] = Field(
        default=None,
        description="Venue specification (nqb or utp_cta)"
    )

# endregion