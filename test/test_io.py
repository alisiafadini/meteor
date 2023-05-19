
import pytest
from   pathlib import Path
from   meteor import io

# todo this is probably the right way to do this
PDB_FILE       = Path(__file__).resolve().parent / 'data' / 'dark.pdb'
PDB_FILE_WRONG = Path(__file__).resolve().parent / 'data' / 'dark-nocryst1.pdb'

def test_get_pdbinfo():

    unit_cell, space_group = io.get_pdbinfo(PDB_FILE)

    assert unit_cell == (51.99, 62.91, 72.03, 90.0, 90.0, 90.0), 'unit cell incorrect'
    assert space_group == 'P212121', 'space group incorrect'


def test_get_pdbinfo_wrong_file():
    # Ensure file with no CRYST1 entry throws an exception
    with pytest.raises(Exception):
        io.get_pdbinfo(PDB_FILE_WRONG)


if __name__ == '__main__':
    test_get_pdbinfo()

