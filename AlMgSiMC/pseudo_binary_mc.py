from cemc.mcmc import PseudoBinarySGC
from ase.clease import CEBulk, Concentration
from cemc import get_atoms_with_ce_calc
from cemc.mcmc import Snapshot, Montecarlo
from ase.clease.tools import wrap_and_sort_by_position
import json
import dataset
import numpy as np
import sys
from ase.io import read
from cemc.tools import FreeEnergy

def get_atoms(cubic=False):
    conc = Concentration(basis_elements=[["Al", "Mg", "Si"]])

    kwargs = {
        "crystalstructure": "fcc",
        "a": 4.05,
        "size": [2, 2, 2],
        "concentration": conc,
        "db_name": "data/almgsi_free_eng.db",
        "max_cluster_size": 4,
        "max_cluster_dia": [7.8, 5.0, 5.0],
        "cubic": cubic
    }
    N = 10
    ceBulk = CEBulk(**kwargs)
    print(ceBulk.basis_functions)
    eci_file = "data/almgsi_fcc_eci.json"
    eci_file = "data/eci_almgsi_aicc.json"
    eci_file = "data/eci_bcs.json"
    eci_file = "data/eci_almgsi_loocv.json"
    eci_file = "data/eci_l1.json"
    with open(eci_file, 'r') as infile:
        ecis = json.load(infile)
    db_name = "large_cell_db{}x{}x{}.db".format(N, N, N)
    atoms = get_atoms_with_ce_calc(ceBulk, kwargs, ecis, size=[N, N, N], 
                                   db_name=db_name)
    return atoms


def free_energy_vs_comp(T, mu, mod):
    from cemc.mcmc import AdaptiveBiasReactionPathSampler
    from cemc.mcmc import ReactionCrdRangeConstraint
    from cemc.mcmc import PseudoBinaryConcObserver
    atoms = get_atoms(cubic=True)
    workdir = "data/pseudo_binary_free"
    mu = float(mu)
    # Have to perform only insert moves
    mc = PseudoBinarySGC(atoms, T, chem_pot=mu,
                         groups=[{"Al": 2}, {"Mg": 1, "Si": 1}],
                         symbols=["Al", "Mg", "Si"], insert_prob=0.1)

    observer = PseudoBinaryConcObserver(mc)
    conc_cnst = ReactionCrdRangeConstraint(observer, value_name="conc")
    conc_cnst.update_range([0, 2000])
    mc.add_constraint(conc_cnst)

    reac_path = AdaptiveBiasReactionPathSampler(
        mc_obj=mc, react_crd=[0.0, 2000], observer=observer,
        n_bins=500, data_file="{}/adaptive_bias{}K_{}mev.h5".format(workdir, T, int(1000*mu)),
        mod_factor=mod, delete_db_if_exists=True, mpicomm=None,
        db_struct="{}/adaptive_bias_struct{}K_{}mev.db".format(workdir, T, int(1000*mu)),
        react_crd_name="conc")

    reac_path.run()
    reac_path.save()


def free_energy_vs_layered(T, mod):
    from cemc.mcmc import AdaptiveBiasReactionPathSampler
    from cemc.mcmc import ReactionCrdRangeConstraint
    from cemc.mcmc import DiffractionObserver
    from ase.geometry import get_layers
    atoms = get_atoms(cubic=True)

    atoms_cpy = atoms.copy()
    layers, dist = get_layers(atoms, (0, 1, 0))
    for atom in atoms_cpy:
        if layers[atom.index] % 2 == 0:
            atom.symbol = "Mg"
        else:
            atom.symbol = "Si"
    symbols = [atom.symbol for atom in atoms_cpy]
    atoms.get_calculator().set_symbols(symbols)

    lamb = 4.05
    k = 2.0*np.pi/lamb
    workdir = "data/diffraction"

    # Have to perform only insert moves
    mc = Montecarlo(atoms, T)
    k_vec = [k, 0, 0]
    observer = DiffractionObserver(atoms=atoms, active_symbols=["Si"], all_symbols=["Mg", "Si"], k_vector=k_vec,
                                   name="reflection")
    conc_cnst = ReactionCrdRangeConstraint(observer, value_name="reflection")
    conc_cnst.update_range([0, 0.5])
    mc.add_constraint(conc_cnst)

    reac_path = AdaptiveBiasReactionPathSampler(
        mc_obj=mc, react_crd=[0.0, 0.5], observer=observer,
        n_bins=50, data_file="{}/layered_bias{}K.h5".format(workdir, T),
        mod_factor=mod, delete_db_if_exists=True, mpicomm=None,
        db_struct="{}/layered_bias_struct{}K.db".format(workdir, T),
        react_crd_name="reflection", ignore_equil_steps=False)
    reac_path.run()
    reac_path.save()


def gs_mgsi():
    atoms = get_atoms(cubic=True)

    symbs = ["Mg" for _ in range(len(atoms))]
    for i in range(int(len(symbs)/2)):
        symbs[i] = "Si"
    atoms.get_calculator().set_symbols(symbs)

    T = [1500, 1400, 1300, 1200, 1100, 1000, 900, 800, 600, 500, 400, 300, 200, 100, 50]
    snap = Snapshot(trajfile="mgsi_gs_search.traj", atoms=atoms)
    for temp in T:
        print("Temperature {}K".format(temp))
        mc = Montecarlo(atoms, temp)
        mc.attach(snap, interval=10*len(atoms))
        mc.runMC(mode="fixed", steps=100*len(atoms), equil=False)


def free_energy(db_name):
    from cemc.tools import FreeEnergy
    db = dataset.connect(db_name)
    tbl = db["simulations"]
    tbl_post = db["post_proc"]
    mu = []
    for row in tbl.find(phase='Al', n_mc_steps=2000000):
        mu.append(row["pseudo_mu"])
    mu = list(set(mu))
    # [{'Al': 1.224744871391589, 'Mg': -1.224744871391589, 'Si': 0.0}, {'Al': -0.7071067811865472, 'Mg': -0.7071067811865472, 'Si': 1.4142135623730951}]
    matrix = np.zeros((3, 3))
    matrix[2, :] = 1.0
    matrix[0, 0] = 1.224744871391589
    matrix[0, 1] = -1.224744871391589
    matrix[0, 2] = 0.0
    matrix[1, 0] = -0.7071067811865472
    matrix[1, 1] = -0.7071067811865472
    matrix[1, 2] = 1.4142135623730951

    for m in mu:
        temps = []
        sgc_energy = []
        ids = []
        singlets = {"c1_0": [], "c1_1": []}
        chem = {"c1_0": None, "c1_1": None}
        for row in tbl.find(phase='Al', pseudo_mu=m):
            temps.append(row["temperature"])
            sgc_energy.append(row["sgc_energy"]/4096)
            ids.append(row["id"])
            singlets["c1_0"].append(row["singlet_c1_0"])
            singlets["c1_1"].append(row["singlet_c1_1"])
            if chem["c1_0"] is None:
                chem["c1_0"] = row["mu_c1_0"]
                chem["c1_1"] = row["mu_c1_1"]
        free = FreeEnergy(limit="lte")
        res = free.free_energy_isochemical(T=temps, sgc_energy=sgc_energy, nelem=3)
        gibbs = free.helmholtz_free_energy(res["free_energy"], singlets, chem)
        
        for i in range(len(gibbs)):
            rhs = np.array([singlets["c1_0"][i], singlets["c1_1"][i], 1.0])
            conc = np.linalg.solve(matrix, rhs)
            entry = {"simID": ids[i], "gibbs": gibbs[i], "grand_pot": res["free_energy"][i],
                     "temperature": res["temperature"][i], "singlet_c1_0": singlets["c1_0"][i],
                     "singlets_c1_1": singlets["c1_1"][i], "al_conc": conc[0], "mg_conc": conc[1], "si_conc": conc[2]}
            tbl_post.upsert(entry, ["simID"])
        #print(res)


def phase_diag():
    from cemc.tools.phasediagram import BinaryPhaseDiagram
    from matplotlib import pyplot as plt
    from ase.units import kB
    from scipy.interpolate import interp1d
    db_name = "sqlite:////work/sophus/almgsi_mc_free_energy_sgc_l1.db"
    db_name = "sqlite:////work/sophus/almgsi_mc_free_energy_sgc_l1_fixed_steps.db"
    db_name = "sqlite:///data/almgsi_mc_free_energy_sgc_l1_fixed_steps.db"

    conc = []
    energy = []
    mu = []
    db = dataset.connect(db_name)
    tbl = db["simulations"]
    for row in tbl.find(phase="random", temperature=2000):
        mu.append(row["pseudo_mu"])
        conc.append(2*row["Al_conc"])
        energy.append(row["sgc_energy"]/4000)
    
    srt_indx = np.argsort(mu).tolist()
    mu = [mu[x] for x in srt_indx]
    energy = [energy[x] for x in srt_indx]
    conc = [conc[x] for x in srt_indx]
    beta_ref = energy[0]
    eng = FreeEnergy()
    ref = eng.free_energy_isothermal(chem_pot=mu, conc=conc, phi_ref=beta_ref)
    G = np.array(ref)
    isochem_ref = {
        "random": interp1d(mu, G/(kB*2000.0), bounds_error=False, fill_value="extrapolate")
    }
    diag = BinaryPhaseDiagram(
                 db_name=db_name, 
                 fig_prefix="data/phasediag_fig/", table="simulations",
                 phase_id="phase", chem_pot="pseudo_mu", energy="sgc_energy",
                 concentration="Al_conc", temp_col="temperature",
                 tol=1E-6, postproc_table="postproc",
                 recalculate_postproc=False, 
                 ht_phases=["random"], num_elem=3,
                 natoms=4000, isochem_ref=isochem_ref,
                 num_per_fu=2)
    mu, T = diag.phase_boundary(phases=["Al", "MgSi"], variable="chem_pot", polyorder=1)
    mu_Al_rnd, T_Al_rnd = diag.phase_boundary(phases=["MgSi", "random_mgsi"], variable="temperature", polyorder=1,
        bounds={"random_mgsi": [1500, 4000]})
    #exit()

    srt_indx = np.argsort(mu).tolist()
    mu = [mu[x] for x in srt_indx]
    T = [T[x] for x in srt_indx]

    srt_indx = np.argsort(mu_Al_rnd).tolist()
    mu_Al_rnd = [mu_Al_rnd[x] for x in srt_indx]
    T_Al_rnd = [T_Al_rnd[x] for x in srt_indx]
    fig_mu = plt.figure()
    ax_mu = fig_mu.add_subplot(1, 1, 1)
    ax_mu.plot(mu, T)
    ax_mu.plot(mu_Al_rnd, T_Al_rnd)
    ax_mu.set_xlabel("Chemical potential (eV)")
    ax_mu.set_ylabel("Temperature (K)")

    print(T)
    conc_al = [np.polyval(diag.composition("Al", temperature=T[i], polyorder=6), mu[i]) for i in range(len(mu))]
    conc_mgsi = [np.polyval(diag.composition("MgSi", temperature=T[i], polyorder=6), mu[i]) for i in range(len(mu))]

    conc_rnd = [np.polyval(diag.composition("MgSi", mu=mu_Al_rnd[i]), T_Al_rnd[i]) for i in range(len(T_Al_rnd))]
    conc_mgsi_rnd = [np.polyval(diag.composition("random_mgsi", mu=mu_Al_rnd[i], bounds=[1400, 5000]), T_Al_rnd[i]) for i in range(len(T_Al_rnd))]
    print(conc_rnd)

    fig_conc, (ax, ax2) = plt.subplots(1, 2, sharey=True)
    ax.plot(conc_mgsi, T, "o", mfc="none")
    ax2.plot(conc_al, T, "o", mfc="none")
    ax.spines["right"].set_visible(False)
    #ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)
    ax2.set_xlim([0.99, 1.0])
    ax.set_ylim([200, 500])
    ax.set_xlim([0, 0.1])
    plt.show()

def free_energy_phase():
    from matplotlib import pyplot as plt
    db_name = "sqlite:///data/almgsi_mc_free_energy_sgc_l1_fixed_steps.db"
    db = dataset.connect(db_name)
    tbl_pp = db["postproc"]
    tbl = db["simulations"]

    ids = []
    T = 310.526315789474
    #T = 478.947368421053

    query = {
        "temperature": {"between": [T-0.1, T+0.1]},
        "phase": "MgSi"
    }
    conc = []
    for row in tbl.find(**query):
        ids.append(row["id"])
        conc.append(row["Al_conc"])
    
    free_eng = []
    for id in ids:
        row = tbl_pp.find_one(systemID=id)
        free_eng.append(row["free_energy"])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(conc, free_eng, "o")
    plt.show()

 
def random_fixed_temp(mu):
    T = 2000
    chem_pots = [mu]
    folder = "/home/gudrun/davidkl/Documents/GPAWTutorial/AlMgSiMC/data"
    fname = folder + "/Al_mgsi_free_eng.xyz"
    atoms_read = wrap_and_sort_by_position(read(fname))
    symbols = [atom.symbol for atom in atoms_read]
    atoms = get_atoms(cubic=True)
    atoms.get_calculator().set_symbols(symbols)
    db_name = "/work/sophus/almgsi_mc_free_energy_sgc_l1_fixed_steps.db"
    for mu in chem_pots:
        print("Current mu: {}".format(mu))
        mc = PseudoBinarySGC(atoms, T, chem_pot=mu,
                            groups=[{"Al": 2}, {"Mg": 1, "Si": 1}],
                            symbols=["Al", "Mg", "Si"])

        nsteps = int(5E5)
        #nsteps = 10000
        mc.runMC(mode="fixed", steps=nsteps, equil_params={"mode": "fixed", "window_length": int(0.1*nsteps)})
        thermo = mc.get_thermodynamic()
        thermo["formula"] = atoms.get_chemical_formula()
        thermo["phase"] = "random"
        thermo["pseudo_mu"] = mu
        db = dataset.connect("sqlite:///{}".format(db_name))
        tbl = db["simulations"]
        tbl.insert(thermo)
        mc.reset_ecis()


def main(argv):
    chem_pot = float(argv[0])
    phase = argv[1]

    folder = "/home/gudrun/davidkl/Documents/GPAWTutorial/AlMgSiMC/data"
    if phase == "Al":
        fname = folder + "/Al_mgsi_free_eng.xyz"
        atoms_read = wrap_and_sort_by_position(read(fname))
        db_energy = -3.724
    else:
        fname = folder + "/MgSi_mgsi_free_eng.xyz"
        atoms_read = read(fname)
        #atoms_read *= (4, 4, 4)
        atoms_read = wrap_and_sort_by_position(atoms_read)
        db_energy = -218.272/64

    symbols = [atom.symbol for atom in atoms_read]


    if phase == "random":
        T = np.linspace(600, 2000, 20)[::-1].tolist()
    elif phase == "random_mgsi":
        T = np.linspace(100, 2000, 20).tolist()
    else:
        T = np.linspace(100.0, 900.0, 20).tolist()
    #temps = [800]
    #T = [300]
    atoms = get_atoms(cubic=True)
    atoms.get_calculator().set_symbols(symbols)
    print(db_energy, atoms.get_calculator().get_energy()/len(atoms))
    #assert abs(db_energy - atoms.get_calculator().get_energy()/len(atoms)) < 1E-3

    db_name = "/work/sophus/almgsi_mc_free_energy_sgc_track_phase.db"
    db_name = "/work/sophus/almgsi_mc_free_energy_sgc_l1_fixed_steps.db"
    for temp in T:
        print("Current temp: {}K".format(temp))
        mc = PseudoBinarySGC(atoms, temp, chem_pot=chem_pot,
                             groups=[{"Al": 2}, {"Mg": 1, "Si": 1}],
                             symbols=["Al", "Mg", "Si"])

        nsteps = int(2E6)
        #nsteps = 10000
        mc.runMC(mode="fixed", steps=nsteps, equil_params={"mode": "fixed", "window_length": int(0.1*nsteps)})
        thermo = mc.get_thermodynamic()
        thermo["formula"] = atoms.get_chemical_formula()
        thermo["phase"] = phase
        thermo["pseudo_mu"] = chem_pot
        num_al = sum(1 for atom in atoms if atom.symbol == "Al")
        num_mg = sum(1 for atom in atoms if atom.symbol == "Mg")
        swapped_phase = False
        if phase == "Al" and num_al < 0.33*len(atoms):
            swapped_phase = True
        if phase == "MgSi" and num_mg < 0.33*len(atoms):
            swapped_phase = True
        if swapped_phase:
            print("Chem. pot {}. Swapped at {}".format(chem_pot, temp))
            return
        db = dataset.connect("sqlite:///{}".format(db_name))
        tbl = db["simulations"]
        tbl.insert(thermo)
        mc.reset_ecis()

if __name__ == "__main__":
    #main(sys.argv[1:])
    #random_fixed_temp(float(sys.argv[1]))
    #free_energy("sqlite:////work/sophus/almgsi_mc_free_energy_sgc_bcs.db")
    #gs_mgsi()
    phase_diag()
    #free_energy_phase()

    run_layered = False
    for arg in sys.argv:
        if arg == "--layered":
            run_layered = True

    if run_layered:
        free_energy_vs_layered(int(sys.argv[1]), float(sys.argv[2]))
    else:
        free_energy_vs_comp(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
