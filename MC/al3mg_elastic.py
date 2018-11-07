from atomtools.ase import ElasticConstants
from ase.io import read
from ase.build import bulk
from ase.visualize import view
import numpy as np
from ase.units import GPa
from cemc.tools import StrainEnergy

DB_NAME = "al3mg_elastic.db"

def prepare():
    atoms = bulk("Al")*(2, 2, 2)
    atoms[4].symbol = "Mg"
    atoms[3].symbol = "Mg"
    factor = 0.5*8.284/4.05
    cell = atoms.get_cell()
    cell *= factor
    atoms.set_cell(cell, scale_atoms=True)
    elastic = ElasticConstants(atoms, DB_NAME)
    # elastic.prepare_db()

def prepare_al():
    atoms = bulk("Al")
    elastic = ElasticConstants(atoms, DB_NAME)
    elastic.prepare_db()

def fit_elastic():
    from ase.db import connect
    atoms = bulk("Al")*(2, 2, 2)
    elastic = ElasticConstants(atoms, DB_NAME)
    db = connect(DB_NAME)
    select_cond = [("formula", "=", "Al"),
                   ("calculator", "=", "gpaw"),
                   ("id", "!=", 25)]
    
    stresses = []
    strains = []
    for row in db.select(select_cond):
        stresses.append(np.array(row["stress"]))
        strain = db.get(id=row.init_struct).data["strain"]
        strains.append(strain)
    C = elastic.get(stresses=stresses, strains=strains, spg=225)
    print("Bulk:")
    print("Voigt: {}".format(elastic.bulk_modulus(mode="V")/GPa))
    print("Reuss: {}".format(elastic.bulk_modulus(mode="R")/GPa))
    print("VRH: {}".format(elastic.bulk_modulus(mode="VRH")/GPa))
    print("Shear:")
    print("Voigt: {}".format(elastic.shear_modulus(mode="V")/GPa))
    print("Reuss: {}".format(elastic.shear_modulus(mode="R")/GPa))
    print("VRH: {}".format(elastic.shear_modulus(mode="VRH")/GPa))
    print("Shear: {}".format(elastic.shear_modulus(mode="VRH")))
    print("Poisson: {}".format(elastic.poisson_ratio))
    #np.set_printoptions(precision=2)
    print(C)

def strain_energy():
    C_al3Mg = np.array([[0.64, 0.24, 0.24, 0,   0,   0],
                        [0.24, 0.64, 0.24, 0,   0,   0],
                        [0.24, 0.24, 0.64, 0,   0,   0],
                        [0,   0,   0,   0.65, 0,   0],
                        [0,   0,   0,   0,   0.65, 0],
                        [0,   0,   0,   0,   0,   0.65]])

    C_al = np.array([[0.63, 0.41, 0.41, 0,   0,   0],
                     [0.41, 0.63, 0.41, 0,   0,   0],
                     [0.41, 0.41, 0.63, 0,   0,   0],
                     [0,   0,   0,   0.43, 0,   0],
                     [0,   0,   0,   0,   0.43, 0],
                     [0,   0,   0,   0,   0,   0.43]])
    C_al = np.array([[0.62639459, 0.41086487, 0.41086487, 0, 0, 0],
            [0.41086487, 0.62639459, 0.41086487, 0, 0, 0],
            [0.41086487, 0.41086487, 0.62639459, 0, 0, 0],
            [0, 0, 0, 0.42750351, 0, 0],
            [0, 0, 0, 0, 0.42750351, 0],
            [0, 0, 0, 0, 0, 0.42750351]])


    eigenstrain = [0.023, 0.023, 0.023, 0.0, 0.0, 0.0]
    mu_al = 0.34876
    mu_al3mg = 0.20835377082353254
    poisson = 0.5*(mu_al + mu_al3mg)
    poisson = mu_al
    strain_energy = StrainEnergy(eigenstrain=eigenstrain, poisson=poisson)
    print(strain_energy.is_isotropic(C_al))
    energy = strain_energy.strain_energy(C_matrix=C_al, C_prec=C_al3Mg)
    v_per_atom = 16.60753125
    print(energy*1000.0*v_per_atom)
    print(energy*1000.0)

if __name__ == "__main__":
    # prepare()
    #fit_elastic()
    # prepare_al()
    strain_energy()