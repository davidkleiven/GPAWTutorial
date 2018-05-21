from ase import Atoms
from ase.visualize import view
import numpy as np
from ase import spacegroup

def view_all():
    a = 4.05
    c = 6.0
    cell = [[2.0*a,0,0],
            [0.0,np.sqrt(3)*a,0.0],
            [0.0,0.0,c]]

    hcp_sites = [
        [a/2.0,0.0,0.0],
        [3.0*a/2.0,0.0,0.0],
        [0.0,np.sqrt(3.0)*a/2.0,0.0],
        [2.0*a,np.sqrt(3.0)*a/2.0,0.0],
        [a/2.0,np.sqrt(3.0)*a,0.0],
        [3.0*a/2.0,np.sqrt(3.0)*a,0.0],
        [a,np.sqrt(3.0)*a/2.0,0.0],
        [a,a*(0.5*np.sqrt(3)-1/np.sqrt(3.0)),c/2.0],
        [a/2.0,0.5*a*(np.sqrt(3)+1.0/np.sqrt(3.0)),c/2.0],
        [3.0*a/2.0,0.5*a*(np.sqrt(3)+1.0/np.sqrt(3.0)),c/2.0]
    ]

    bcc_sites = [
        [0.0,0.0,0.0],
        [2.0*a,0.0,0.0],
        [0.0,np.sqrt(3.0)*a,0.0],
        [2.0*a,np.sqrt(3.0)*a,0.0],
        [a,0.5*np.sqrt(3)*a,c/2.0]
    ]

    hcp_symbs = ["V" for _ in range(len(hcp_sites))]
    bcc_symbs = ["Si" for _ in range(len(bcc_sites))]
    symbs = hcp_symbs+bcc_symbs
    sites = hcp_sites+bcc_sites
    atoms = Atoms(symbs,sites)
    atoms.set_cell(cell)
    view(atoms)

def find_spacegroup():
    a = 4.05
    c = 6.0
    cell = [[2.0*a,0,0],
            [0.0,np.sqrt(3)*a,0.0],
            [0.0,0.0,c]]

    hcp_sites = [
        [a/2.0,0.0,0.0],
        [3.0*a/2.0,0.0,0.0],
        [0.0,np.sqrt(3.0)*a/2.0,0.0],
        #[2.0*a,np.sqrt(3.0)*a/2.0,0.0],
        #[a/2.0,np.sqrt(3.0)*a,0.0],
        #[3.0*a/2.0,np.sqrt(3.0)*a,0.0],
        [a,np.sqrt(3.0)*a/2.0,0.0],
        [a,a*(0.5*np.sqrt(3)-1/np.sqrt(3.0)),c/2.0],
        [a/2.0,0.5*a*(np.sqrt(3)+1.0/np.sqrt(3.0)),c/2.0],
        [3.0*a/2.0,0.5*a*(np.sqrt(3)+1.0/np.sqrt(3.0)),c/2.0]
    ]

    bcc_sites = [
        [0.0,0.0,0.0],
        #[2.0*a,0.0,0.0],
        #[0.0,np.sqrt(3.0)*a,0.0],
        #[2.0*a,np.sqrt(3.0)*a,0.0],
        [a,0.5*np.sqrt(3)*a,c/2.0]
    ]

    hcp_symbs = ["V" for _ in range(len(hcp_sites))]
    bcc_symbs = ["Si" for _ in range(len(bcc_sites))]
    symbs = hcp_symbs+bcc_symbs
    sites = hcp_sites+bcc_sites
    atoms = Atoms(symbs,sites)
    atoms.set_cell(cell)
    view(atoms)

    # Find spacegroup
    sp = spacegroup.get_spacegroup(atoms)
    print (sp)
    
find_spacegroup()
