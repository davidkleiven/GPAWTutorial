#from atomtools.ase import ElasticConstants
from ase.io import read
import numpy as np
from ase.build import bulk
from ase.units import GPa
import matplotlib as mpl
mpl.rcParams.update({"svg.fonttype": "none", "font.size": 18,
                     "axes.unicode_minus": False})

DB_NAME = "elastic_mgsi100.db"

def prepare_mgsi():
    atoms = read("data/mgsi100_fully_relaxed.xyz")
    elastic = ElasticConstants(atoms, DB_NAME)
    # elastic.prepare_db()

def fit_elastic():
    from ase.db import connect
    atoms = bulk("Al")*(2, 2, 2)
    elastic = ElasticConstants(atoms, DB_NAME)
    db = connect(DB_NAME)
    
    stresses = []
    strains = []
    for row in db.select('id>=25'):
        stress = np.array(row["stress"])
        stresses.append(stress)
        strain = np.array(db.get(id=row.init_struct).data["strain"])
        strains.append(strain)

    from ase.spacegroup import Spacegroup
    spg = Spacegroup(123)
    print(spg.get_rotations())
    C = elastic.get(stresses=stresses, strains=strains, spg=225, perm="zxy")
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
    np.savetxt("data/C_MgSi100_225.csv", C, delimiter=",")

def misfit_strain():
    from ase.spacegroup import get_spacegroup
    mgsi100 = read("data/mgsi100_fully_relaxed.xyz")
    print(get_spacegroup(mgsi100, symprec=1E-4))
    atoms = bulk("Al")*(2, 2, 2)
    cell_mgsi = mgsi100.get_cell().T
    cell_al = atoms.get_cell().T
    F = cell_mgsi.dot(np.linalg.inv(cell_al))
    eps = 0.5*(F.T.dot(F) - np.eye(3))
    print(eps)
    print(np.linalg.eigvals(eps))

def explore_orientations():
    from cemc.tools import StrainEnergy
    from matplotlib import pyplot as plt
    mu = 0.27155342515650893
    C_al = C_al = np.array([[0.62639459, 0.41086487, 0.41086487, 0, 0, 0],
            [0.41086487, 0.62639459, 0.41086487, 0, 0, 0],
            [0.41086487, 0.41086487, 0.62639459, 0, 0, 0],
            [0, 0, 0, 0.42750351, 0, 0],
            [0, 0, 0, 0, 0.42750351, 0],
            [0, 0, 0, 0, 0, 0.42750351]])
    C_al = np.loadtxt("data/C_al.csv", delimiter=",")
    C_mgsi = np.loadtxt("data/C_MgSi100_225.csv", delimiter=",")
    misfit = np.array([[ 0.0440222,   0.00029263,  0.0008603 ],
              [ 0.00029263, -0.0281846,   0.00029263],
              [ 0.0008603,   0.00029263,  0.0440222 ]])


    strain_eng = StrainEnergy(poisson=mu, misfit=misfit)
    ellipsoid = {
        "aspect": [10.0, 10.0, 1.0],
        "C_prec": C_mgsi
        #"scale_factor": 0.8
    }
    res = strain_eng.explore_orientations(
        ellipsoid, C_al, step=5, phi_ax="x")
    strain_eng.plot_explore_result(res)
    plt.show()

def get_voxels(N, prec_type="plate"):
    voxels = np.zeros((N, N, N), dtype=np.uint8)
    if prec_type == "plate":
        voxels[:, :, :2] = 1
    elif prec_type == "needle":
        voxels[:, :2, :2] = 1
    return voxels

def explore_khachaturyan(prec_type="plate"):
    from cemc.tools import Khachaturyan
    C_al = np.loadtxt("data/C_al.csv", delimiter=",")
    #C_mgsi = np.loadtxt("data/C_MgSi100_225.csv", delimiter=",")
    misfit = np.array([[ 0.0440222,   0.00029263,  0.0008603 ],
              [ 0.00029263, -0.0281846,   0.00029263],
              [ 0.0008603,   0.00029263,  0.0440222 ]])
    
    strain = Khachaturyan(misfit_strain=misfit, elastic_tensor=C_al)
    voxels = get_voxels(256, prec_type=prec_type)
    fname = "data/strain_energy_{}.csv".format(prec_type)
    strain.explore_orientations(voxels, fname=fname, step=22)
    


if __name__ == "__main__":
    # prepare_mgsi()
    #fit_elastic()
    #misfit_strain()
    #explore_orientations()
    explore_khachaturyan(prec_type="needle")