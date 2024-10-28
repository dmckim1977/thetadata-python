# """Module for handling API data pagination and parsing."""
# from functools import wraps
# from typing import Any, Callable, Dict, Optional
#
# import httpx
# import pandas as pd
#
# from thetadata import parse_trade
#
#
# class DataHandler:
#     """Handles data retrieval, pagination, and parsing for ThetaClient."""
#
#     def __init__(self, base_url: str, parser: Callable):
#         """
#         Initialize the DataHandler.
#
#         Args:
#             base_url: Base URL for API requests
#             parser: Function to parse individual response pages
#         """
#         self.base_url = base_url
#         self.parser = parser
#
#     def get_paginated_data(
#             self,
#             endpoint: str,
#             params: Dict[str, Any],
#             timeout: Optional[int] = None
#     ) -> pd.DataFrame:
#         """
#         Retrieve and combine all pages of data from the API.
#
#         Args:
#             endpoint: API endpoint to query
#             params: Query parameters for initial request
#             timeout: Request timeout in seconds
#
#         Returns:
#             Combined DataFrame of all pages
#         """
#         url = f"{self.base_url}{endpoint}"
#         dataframes = []
#         current_params = params
#
#         while True:
#             # Make request for current page
#             response = httpx.get(
#                 url,
#                 params=current_params,
#                 timeout=timeout
#             ).raise_for_status().json()
#
#             # Parse the current page
#             df = self.parser(response)
#             dataframes.append(df)
#
#             # Check if there are more pages
#             next_page = response.get('header', {}).get('next_page')
#             if next_page == 'null' or not next_page:
#                 break
#
#             # Update URL for next page and clear params
#             url = next_page
#             current_params = None
#
#         # Combine all dataframes
#         if len(dataframes) == 1:
#             return dataframes[0]
#
#         return pd.concat(dataframes, axis=0, ignore_index=True)
#
#
# def with_pagination(func):
#     """
#     Decorator to handle pagination for ThetaClient methods.
#
#     Automatically uses DataHandler to handle pagination for any client method
#     that returns paginated data.
#     """
#
#     @wraps(func)
#     def wrapper(self, *args, **kwargs):
#         # Get the endpoint and params from the decorated method
#         endpoint, params = func(self, *args, **kwargs)
#
#         # Create DataHandler instance with appropriate parser
#         handler = DataHandler(
#             base_url=f"http://{self.host}:{self.port}",
#             parser=parse_trade
#         )
#
#         # Get paginated data
#         return handler.get_paginated_data(endpoint, params)
#
#     return wrapper
