from pydantic import Field

from .base import DateRange


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