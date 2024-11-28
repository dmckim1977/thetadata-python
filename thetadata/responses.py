# Response model
from typing import List

from pydantic import BaseModel


class OHLCV(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class StockHistoricalEODReport(BaseModel):
    trades: List[TradeData]
    symbol: str
    market: str