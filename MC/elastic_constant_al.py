"""Compute the elastic constants of some important calculation."""
import gpaw as gp
from ase.build import bulk
from atomtools.ase import ElasticConstants
from ase.units import GPa
from ase.io import read
from cemc.tools import StrainEnergy
from cemc.tools import to_mandel, to_full_tensor, rotate_tensor, rot_matrix
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend("tkAgg")

prepare = False
run = False
postproc = True
eigenstrain = False
atoms = bulk("Al")
db_name = "elastic_constant_al.db"
atoms = read("data/relaxed_mgsi.xyz")
el = ElasticConstants(atoms, db_name)
if prepare:
    el.prepare_db()
elif run:
    calc = gp.GPAW(mode=gp.PW(600), xc="PBE", kpts=(8, 8, 8), nbands=-100)
    atoms.set_calculator(calc)
    for uid in range(1, 25):
        el.run(uid, calc)
elif postproc:
    C = el.get(spg=225)
    np.savetxt("data/C_al.csv", C, delimiter=",")
    mode = "V"
    B = el.bulk_modulus(mode=mode)/GPa
    G = el.shear_modulus(mode=mode)/GPa
    E = el.youngs_modulus(mode=mode)/GPa
    mu = el.poisson_ratio

    print("Bulk modulus: {} GPa".format(B))
    print("Shear modulus: {} GPa".format(G))
    print("Poisson ratio: {}".format(mu))
    print("Young's modulus {} GPa".format(E))

    eigenstrain_princ = [-0.06497269, -0.06497269, 0.09606948, 0.0, 0.0, 0.0]
    #eigenstrain_princ = [-0.06497269, -0.06497269, 0.04, 0.0, 0.0, 0.0]
    eigenstrain = [-0.01129197, -0.01129197, -0.01129197, 0.05368072,
                   0.05368072, 0.05368072]
    strain_eng = StrainEnergy(poisson=mu, eigenstrain=eigenstrain)
    ellipsoid = {
        "aspect": [1000.0, 1.0, 1.0],
        "scale_factor": 0.83
    }
    # strain_eng.explore_aspect_ratios(0.83, C)
    res = strain_eng.explore_orientations(ellipsoid, C, step=5)
    # opt_res = strain_eng.optimize_rotation(ellipsoid, C, res[0]["rot_seq"])
    # strain_eng.show_ellipsoid(ellipsoid, res[0]["rot_seq"])
    strain_eng.plot_explore_result(res)
    # print(opt_res)
    # fig = strain_eng.plot(0.83, C, rot_seq=res[0]["rot_seq"])
    # fig = strain_eng.plot(0.83, C, rot_seq=res[0]["rot_seq"], latex=True)
    plt.show()
elif eigenstrain:
    ref_atoms = bulk("Al")
    ref_atoms *= (2, 2, 2)
    ref_cell = ref_atoms.get_cell().T
    strained_cell = atoms.get_cell().T
    princ_strain = ElasticConstants.get_strain(ref_cell, strained_cell,
                                               principal=False)
    print(princ_strain)
