import matplotlib as mpl
mpl.use('Agg')
from ase.db import connect
from hiphive.structure_generation import generate_mc_rattled_structures
from ase.calculators.emt import EMT
from ase.io import read
from hiphive.utilities import prepare_structures
from ase.visualize import view
import random
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.fitting import Optimizer
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from ase import Atoms
from matplotlib import pyplot as plt
import numpy as np
from hiphive.calculators import ForceConstantCalculator
from ase.build import make_supercell
from hiphive.core.translational_constraints import get_translational_constraint_matrix
from hiphive.core.rotational_constraints import get_rotational_constraint_matrix

ref_fcc = "data/mgsi100_fully_relaxed.xyz"
ref_fcc_conv = "data/mgsi_rattled8full_relax.xyz"
beta_struct = "data/mgsi_interstitial_fully_relaxed.xyz"
ref_fcc_conv2x2x2 = "data/mgsi2.xyz"

DB_NAME = "mgsi_hiphive.db"
rattle_std = 0.01
minimum_distance = 2.3
hbar = 4.135667696e-15  # eVs


def prepare(phase):
    db = connect(DB_NAME)

    if phase == 'fcc':
        ref_struct = read(ref_fcc)*(2, 2, 2)
    elif phase == 'fcc_conventional':
        ref_struct = read(ref_fcc_conv)*(2, 2, 2)
    elif phase == 'beta':
        ref_struct = read(beta_struct)*(2, 2, 2)
    elif phase == 'fcc2x2x2':
        ref_struct = read(ref_fcc_conv2x2x2)

    db.write(ref_struct, group=3, comment="Reference structure for mgfi FCC conv")
    structures = generate_mc_rattled_structures(ref_struct, 20, rattle_std,
                                                minimum_distance)
    for s in structures:
        num = random.randint(0, 2**32-1)
        db.write(s, group=num, phase=phase, rattle_std=rattle_std)


def fit(phase):
    db = connect(DB_NAME)
    if phase == 'fcc':
        ref_fcc = db.get(id=1).toatoms()
    elif phase == 'fcc2x2x2':
        ref_fcc = db.get(group=3).toatoms()
    else:
        ref_fcc = db.get(id=41)
    structures = []
    groups = []
    scond = [('phase', '=', phase), ('rattle_std','<',0.02)]
    for row in db.select(scond):
        groups.append(row.group)

    for g in groups:
        try:
            atoms = db.get(group=g, struct_type='final').toatoms()
            structures.append(atoms)
        except Exception as exc:
            print(exc)

    structures = prepare_structures(structures, ref_fcc)

    cutoffs = [3.0, 3.0, 3.0]
    cs = ClusterSpace(structures[0], cutoffs, symprec=1e-4)
    print(cs)
    print(structures)

    sc = StructureContainer(cs)
    for structure in structures:
        try:
            sc.add_structure(structure)
        except Exception as exc:
            print(exc)
    print(sc)

    A, y = sc.get_fit_data()
    At = get_rotational_constraint_matrix(sc.cluster_space)
    yt = np.zeros((At.shape[0]))
    lam_trans = 0.001
    A_full = np.vstack((A, lam_trans * At))
    y_full = np.hstack((y, yt))
    # A_full = A
    # y_full = y
    

    opt = Optimizer((A_full, y_full), standardize=False)
    opt.train()
    #plot_fit(opt, sc)

    print(opt)

    fcp = ForceConstantPotential(cs, opt.parameters)
    fcp.write(f'mgsi_{phase}.fcp')
    print(fcp)


def plot_fit(opt, sc):
    X, y = sc.get_fit_data()
    pred_train = opt.predict(X[opt.train_set, :])
    pred_test = opt.predict(X[opt.test_set, :])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(pred_train, y[opt.train_set], 'o', mfc='none', label="Train")
    ax.plot(pred_test, y[opt.test_set], 'o', mfc='none', label="Test")
    minval = np.min(y)
    maxval = np.max(y)
    rng = maxval - minval
    ax.plot([minval-0.05*rng, maxval+0.05*rng],
            [minval-0.05*rng, maxval+0.05*rng])
    ax.set_xlabel("Predicted (eV/A)")
    ax.set_ylabel("DFT force (eV/A)")
    ax.legend()
    plt.savefig("fitForces.png", dpi=200)
    plt.show()



def get_band(q_start, q_stop, N):
    """ Return path between q_start and q_stop """
    return np.array([q_start + (q_stop-q_start)*i/(N-1) for i in range(N)])


def md_calculation(fcp_file):
    from ase import units
    from ase.io.trajectory import Trajectory
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.langevin import Langevin
    from ase.md import MDLogger

    if 'fcc' in fcp_file:
        ref_cell = read(ref_fcc)
        P = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        P *= 3
        atoms = make_supercell(ref_cell, P)
    else:
        raise ValueError("Unknown phase")

    fcp = ForceConstantPotential.read(fcp_file)
    fcs = fcp.get_force_constants(atoms)
    calc = ForceConstantCalculator(fcs)
    atoms.set_calculator(calc)

    temperature = 200
    number_of_MD_steps = 100
    time_step = 5  # in fs
    dump_interval = 10
    traj_file = "data/md_" + fcp_file.split('.')[0] + '{}.traj'.format(temperature)
    log_file = "data/md_log_{}.log".format(temperature)

    dyn = Langevin(atoms, time_step * units.fs, temperature * units.kB, 0.02)
    logger = MDLogger(dyn, atoms, log_file,
                      header=True, stress=False, peratom=True, mode='w')
    traj_writer = Trajectory(traj_file, 'w', atoms)
    dyn.attach(logger, interval=dump_interval)
    dyn.attach(traj_writer.write, interval=dump_interval)

    # run MD
    MaxwellBoltzmannDistribution(atoms, temperature * units.kB)
    dyn.run(number_of_MD_steps)


def phonon_dos(fcp_file):
    if 'fcc2x2x2' in fcp_file:
        prim = read(ref_fcc_conv2x2x2)
    else:
        prim = read(ref_fcc)
    fcp = ForceConstantPotential.read(fcp_file)
    mesh = [33, 33, 33]

    atoms_phonopy = PhonopyAtoms(symbols=prim.get_chemical_symbols(),
                                 scaled_positions=prim.get_scaled_positions(),
                                 cell=prim.cell)

    phonopy = Phonopy(atoms_phonopy, supercell_matrix=5*np.eye(3),
                      primitive_matrix=None)

    supercell = phonopy.get_supercell()
    supercell = Atoms(cell=supercell.cell, numbers=supercell.numbers, pbc=True,
                      scaled_positions=supercell.get_scaled_positions())
    fcs = fcp.get_force_constants(supercell)

    phonopy.set_force_constants(fcs.get_fc_array(order=2))
    phonopy.set_mesh(mesh, is_eigenvectors=True, is_mesh_symmetry=False)
    phonopy.run_total_dos()
    phonopy.plot_total_DOS() 
    plt.savefig("phononDOS.png", dpi=200)

    Nq = 51
    G2X = get_band(np.array([0, 0, 0]), np.array([0.5, 0.5, 0]), Nq)
    X2K2G = get_band(np.array([0.5, 0.5, 1.0]), np.array([0, 0, 0]), Nq)
    G2L = get_band(np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5]), Nq)
    bands = [G2X, X2K2G, G2L]

    phonopy.set_band_structure(bands)
    phonopy.plot_band_structure()
    xticks = plt.gca().get_xticks()
    xticks = [x*hbar*1e15 for x in xticks]  # Convert THz to meV
    # plt.gca().set_xticks(xticks)
    plt.gca().set_xlabel("Frequency (THz)")
    plt.savefig("phononBand.png", dpi=200)
    phonopy.run_thermal_properties(t_step=10, t_max=800, t_min=100)
    tp_dict = phonopy.get_thermal_properties_dict()
    temperatures = tp_dict['temperatures']
    free_energy = tp_dict['free_energy']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(temperatures, free_energy)

    plt.show()


fit("fcc2x2x2")
#phonon_dos('mgsi_fcc2x2x2.fcp')
#md_calculation('mgsi_fcc.fcp')
#prepare_fcc()
#prepare("beta")
#prepare("fcc")
#prepare("fcc2x2x2")
