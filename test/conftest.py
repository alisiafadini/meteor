from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parent.resolve() / "data"


@pytest.fixture(scope="session")
def example_pdb_file(data_dir: Path) -> Path:
    return data_dir / "rsEGFP2_dark_no-chromophore.pdb"
