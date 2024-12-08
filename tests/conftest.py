# tests/conftest.py
# conftest.py
import os
from typing import Generator
import pytest
from dotenv import load_dotenv

from thetadata import ThetaClient  # Make sure this import works


def pytest_configure(config):
    """Load environment variables before running tests."""
    load_dotenv()

    # Verify required environment variables
    required_vars = ['THETAUSER', 'THETAPASS']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        pytest.exit(
            f"Missing required environment variables: {', '.join(missing_vars)}")


@pytest.fixture(scope="session")
def theta_credentials() -> tuple[str, str]:
    """Get Thetadata credentials from environment."""
    username = os.getenv('THETAUSER')
    password = os.getenv('THETAPASS')
    return username, password


@pytest.fixture(scope="session")
def theta_client(theta_credentials) -> Generator[ThetaClient, None, None]:
    """Create a ThetaClient instance for testing."""
    username, password = theta_credentials
    client = ThetaClient(username=username, passwd=password)

    yield client

    # Cleanup after all tests
    client.cleanup()





#
# def pytest_sessionstart(session):
#     """Load environment variables at session start"""
#     load_dotenv()
#     THETAPASS=os.environ['THETAPASS']
#     THETAUSER=os.environ['THETAUSER']
#     return THETAPASS, THETAUSER
#
#
# @pytest.fixture(scope="session")
# def env_vars():
#     """Fixture for environment variables"""
#     load_dotenv()
#     THETAPASS = os.environ['THETAPASS']
#     THETAUSER = os.environ['THETAUSER']
#
#     required_vars = [
#         "THETAPASS",
#         "THETAUSER"
#     ]
#
#     missing_vars = [var for var in required_vars if not os.getenv(var)]
#     if missing_vars:
#         raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
#
#     return {var: os.getenv(var) for var in required_vars}
#
#
# @pytest.fixture(scope="session")
# def theta_client(env_vars):
#     """
#     Create a shared ThetaClient instance for all tests.
#     Scope is set to 'session' so the same client is reused across all tests.
#     """
#     client = ThetaClient(
#         username=env_vars["THETAUSER"],
#         passwd=env_vars["THETAPASS"]
#     )
#     yield client  # using yield allows for cleanup after all tests complete
#     # Add any cleanup here if needed
#     # This code runs after all tests finish

#
# # tests/test_expirations.py
# def test_get_expirations(theta_client):
#     with theta_client.connect():
#         expirations = theta_client.expirations("AAPL")
#         assert expirations is not None
#     # Add more specific assertions




