from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from .base import DateRange, RootMixin


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


class ExpirationsRequest(RootMixin):
    """Request parameters for getting option expirations.

    :param root: The root/ticker symbol to get expirations for
    """
    pass  # Inherits root validation from RootMixin


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