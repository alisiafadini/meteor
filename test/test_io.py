
from meteor import io

# todo this is probably the right way to do this
#PDB_FILE = pathlib.Path(__file__) / 'data' / 'dark.pdb'
PDB_FILE = './data/dark.pdb'


def test_get_pdbinfo():

    unit_cell, space_group = io.get_pdbinfo(PDB_FILE)

    assert unit_cell == [51.99, 62.91, 72.03, 90.0, 90.0, 90.0], 'unit cell incorrect'
    assert space_group == 'P212121', 'space group incorrect'

    # todo write a file without CRYST1
    # check to make sure that fails

    return


if __name__ == '__main__':
    test_get_pdbinfo()

