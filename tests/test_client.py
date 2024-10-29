"""
Test for the ThetaTerminal.

- Test Terminal with launch() and kill() on initialization.
- Test Terminal connect() context manager.

"""
import logging
import os
import pytest
from dotenv import load_dotenv
from thetadata import ThetaClient


@pytest.fixture(scope="session")
def env_variables():
    """Fixture to load environment variables"""
    load_dotenv()
    credentials = {
        'username': os.getenv('THETAUSER'),
        'password': os.getenv('THETAPASS')
    }

    if not all(credentials.values()):
        pytest.fail(
            "Environment variables THETAUSER and THETAPASS must be set")

    return credentials


@pytest.fixture(scope="function")
def theta_client(env_variables):
    """Fixture to create and cleanup a ThetaClient instance"""
    client = ThetaClient(
        username=env_variables['username'],
        passwd=env_variables['password'],
        launch=True
    )

    yield client

    # Cleanup after each test
    client.kill()


def test_client_initialization(env_variables):
    """Test that client initializes successfully"""
    client = ThetaClient(
        username=env_variables['username'],
        passwd=env_variables['password'],
        launch=True
    )

    try:
        assert isinstance(client, ThetaClient)
    finally:
        client.kill()


def test_client_status(theta_client):
    """Test that client can check status"""
    status = theta_client.status()
    assert status is not None


def test_client_lifecycle(env_variables):
    """Test the full lifecycle of client creation, status check, and cleanup"""
    client = ThetaClient(
        username=env_variables['username'],
        passwd=env_variables['password'],
        launch=True
    )

    try:
        # Test initial status
        initial_status = client.status()
        assert initial_status is not None

        # Test kill operation
        client.kill()
        # Note: You might want to add additional assertions here
        # depending on how the client behaves after being killed
    finally:
        # Ensure cleanup even if test fails
        try:
            client.kill()
        except Exception as e:
            logging.error('Failed to kill client. Error: {}'.format(e))
