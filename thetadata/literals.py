from datetime import date
from typing import Literal, get_args


# region Define Literals
SecurityType = Literal["option", "stock", "index"]
OptionReqType = Literal["quote", "trade", "implied_volatility"]
OptionRight = Literal['C', 'P']
Terminal = Literal['MDDS', 'FPSS']
Rate = Literal['SOFR', 'TREASURY_M1', 'TREASURY_M3', 'TREASURY_M6',
    'TREASURY_Y1', 'TREASURY_Y2', 'TREASURY_Y3', 'TREASURY_Y5', 'TREASURY_Y7',
    'TREASURY_Y10', 'TREASURY_Y20', 'TREASURY_Y30']

def is_security_type(value: str):  # TODO error checking here
    return value in get_args(SecurityType)


def is_right(value: str):  # TODO error checking here
    return value in get_args(OptionRight)


def is_option_req(value: str):  # TODO error checking here
    return value in get_args(OptionReqType)

# endregion