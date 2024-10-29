import logging
from datetime import date, time as time


def ms_to_time(ms: int) -> time:
    """Convert milliseconds since midnight to time object.

    """
    try:
        seconds = ms // 1000
        microseconds = (ms % 1000) * 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return time(hour=hours, minute=minutes, second=seconds,
                    microsecond=microseconds)
    except Exception as e:
        logging.error(f"Failed to convert milliseconds: {ms} to time. {e}")
        return time(hour=0, minute=0, second=0, microsecond=0)


def time_to_ms(time_str: str) -> int:
    """

    """
    parts = time_str.split(':')

    if len(parts) == 3:  # H:M:S format
        hours, minutes, seconds = map(int, parts)
    elif len(parts) == 2:  # H:M format
        hours, minutes = map(int, parts)
        seconds = 0
    else:
        raise ValueError("Invalid time format. Use H:M:S or H:M")

    # Convert to milliseconds
    total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000
    logging.info(total_ms)
    return total_ms


def _format_strike(strike: float) -> int:
    """Round USD to the nearest tenth of a cent, acceptable by the terminal."""
    return round(strike * 1000)


def _format_date(dt: date) -> str:
    """Format a date obj into a string acceptable by the terminal."""
    return dt.strftime("%Y%m%d")
