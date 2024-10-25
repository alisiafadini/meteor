from pathlib import Path

import pytest

from meteor.testing import check_test_file_exists


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parent.resolve() / "data"


@pytest.fixture(scope="session")
def example_pdb_file(data_dir: Path) -> Path:
    return data_dir / "8a6g.pdb"


@pytest.fixture(scope="session")
def testing_mtz(data_dir: Path) -> Path:
    path = data_dir / "scaled-test-data.mtz"
    check_test_file_exists(path)
    return path
