#from atomtools.ase import ElasticConstants
from ase.io import read
import numpy as np
from ase.build import bulk
from ase.units import GPa
import matplotlib as mpl
from ase.db import connect
from atomtools.eos import BirschMurnagan
mpl.rcParams.update({"svg.fonttype": "none", "font.size": 18,
                     "axes.unicode_minus": False})

DB_NAME = "elastic_mgsi_interstitial.db"

def prepare_mgsi_interstitial():
    atoms = read("data/mgsi_interstitial_fully_relaxed.xyz")
    elastic = ElasticConstants(atoms, DB_NAME)
    #elastic.prepare_db()

def prepare_db_bulk_mod_fit():
    mgsi_interstitial = read("data/mgsi_interstitial_fully_relaxed.xyz")
    scale = np.linspace(0.95, 1.05, 20)
    db_name = "bulk_mod_fit_beta_eye.db"
    db = connect(db_name)
    for i in range(len(scale)):
        atoms = mgsi_interstitial.copy()
        cell = atoms.get_cell()*scale[i]
        atoms.set_cell(cell, scale_atoms=True)
        db.write(atoms)

def birsch_murnag():
    from ase.units import GPa
    from matplotlib import pyplot as plt
    db_name = "bulk_mod_fit_beta_eye.db"
    db = connect(db_name)
    V = []
    E = []
    for row in db.select(calculator="gpaw"):
        E.append(row.energy)
        V.append(row.toatoms().get_volume())

    eos = BirschMurnagan(np.array(V), np.array(E))
    eos.fit()
    eos.plot()
    plt.show()
    vol = V[np.argmin(E)]
    print(eos.bulk_modulus(vol)/GPa)

def misfit_strain():
    from ase.spacegroup import get_spacegroup
    mgsi_interstitial = read("data/mgsi_interstitial_fully_relaxed.xyz")
    print(get_spacegroup(mgsi_interstitial, symprec=1E-4))
    atoms = bulk("Al", crystalstructure="sc", cubic=True, a=2.025)*(4, 4, 2)
    cell_mgsi = mgsi_interstitial.get_cell().T
    cell_al = atoms.get_cell().T
    F = cell_mgsi.dot(np.linalg.inv(cell_al))
    eps = 0.5*(F.T.dot(F) - np.eye(3))
    print(eps)
    print(np.linalg.eigvals(eps))

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
    C = elastic.get(stresses=stresses, strains=strains, spg=123)
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
    np.savetxt("data/C_beta_eye.csv", C, delimiter=",")

def explore_orientations():
    from cemc.tools import StrainEnergy
    from matplotlib import pyplot as plt
    mu = 0.27155342515650893
    C_al = np.loadtxt("data/C_al.csv", delimiter=",")
    C_mgsi = np.loadtxt("data/C_beta_eye.csv", delimiter=",")
    misfit = np.array([[ 6.31232839e-02,  2.96796703e-07,  0.00000000e+00],
                       [ 2.96796703e-07,  6.31232839e-02,  0.00000000e+00],
                       [ 0.00000000e+00,  0.00000000e+00, -6.80589135e-05]])


    strain_eng = StrainEnergy(poisson=mu, misfit=misfit)
    ellipsoid = {
        "aspect": [10000.0, 1.0, 1.0],
        "C_prec": C_mgsi
        #"scale_factor": 0.8
    }
    res = strain_eng.explore_orientations(
        ellipsoid, C_al, step=5, phi_ax="z")
    strain_eng.plot_explore_result(res)
    plt.show()

def get_voxels(N, prec_type="plate"):
    voxels = np.zeros((N, N, N), dtype=np.uint8)
    width = int(N/4)
    if prec_type == "plate":
        voxels[:width, :width, :2] = 1
    elif prec_type == "needle":
        voxels[:width, :2, :2] = 1
    return voxels

def explore_khachaturyan(prec_type="plate"):
    from cemc.tools import Khachaturyan
    C_al = np.loadtxt("data/C_al.csv", delimiter=",")
    #C_mgsi = np.loadtxt("data/C_MgSi100_225.csv", delimiter=",")
    
    misfit = np.array([[ 6.31232839e-02,  2.96796703e-07,  0.00000000e+00],
                       [ 2.96796703e-07,  6.31232839e-02,  0.00000000e+00],
                       [ 0.00000000e+00,  0.00000000e+00, -6.80589135e-05]])
    strain = Khachaturyan(misfit_strain=misfit, elastic_tensor=C_al)
    if prec_type == "plate":
        phi_ax = "x"
    else:
        phi_ax = "z"
    voxels = get_voxels(128, prec_type=prec_type)
    fname = "data/strain_energy_interstitial{}.csv".format(prec_type)
    strain.explore_orientations(voxels, fname=fname, step=5, phi_ax=phi_ax)

if __name__ == "__main__":
    #prepare_mgsi_interstitial()
    # misfit_strain()
    #fit_elastic()
    #prepare_db_bulk_mod_fit()
    #explore_orientations()
    #birsch_murnag()
    explore_khachaturyan(prec_type="needle")
