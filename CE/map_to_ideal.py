from ase.db import connect
from scipy.spatial import cKDTree as KDTree
import numpy as np

def map_to_ideal( db_name, atoms ):
    """
    Maps a relaxed structure to the ideal structure
    """
    db = connect( db_name )
    row = db.get( name="information" )
    template = row.toatoms()

    if ( len(atoms) != len(template) ):
        raise ValueError( "The size of the atoms object has to be the same as the template" )

    atoms_cpy = atoms.copy()

    # Set the cell equal to the template
    atoms_cpy.set_cell( template.get_cell(), scale_atoms=True )

    ref_positions = template.get_positions()

    tree = KDTree( ref_positions )

    source_positions = atoms_cpy.get_positions()
    dist, indx = tree.query( source_positions )

    # Check that no index appears twice
    if ( len(np.unique(indx)) != len(indx) ):
        msg = "Could not find the mapping from the relaxed to the unrelaxed. "
        msg += "At least two atoms are going to be mapped onto the same."
        raise ValueError( msg)

    for i in range(len(indx)):
        template[indx[i]].symbol = atoms_cpy[i].symbol
    return template
