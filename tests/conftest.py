import pytest

from carseats_ml import data


@pytest.fixture(scope="session")
def prepared():
    """The scaled train/test splits, prepared once for the whole test session."""
    return data.prepare()
