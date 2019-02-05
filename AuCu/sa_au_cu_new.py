import dataset
from cemc.mcmc import Montecarlo, EnergyEvolution
from cemc import get_atoms_with_ce_calc
import sys
import json
from ase.clease import CEBulk, Concentration
from ase.io import write
import numpy as np
from ase.io import read
from ase.clease.tools import wrap_and_sort_by_position


def sa(au_comp, kwargs, eci, db_name, size):
    bc = CEBulk(**kwargs)
    atoms = get_atoms_with_ce_calc(bc, kwargs, eci, size, db_name="cu-au_quad.db")
    temperatures = [1500, 1400, 1300, 1200, 1100, 1000, 900, 800, 700, 600,
                    500, 400, 300, 200, 100, 50, 25, 10]
    N = len(atoms)

    gs = wrap_and_sort_by_position(read("data/atoms_Au250Cu750.xyz"))
    symbs = [atom.symbol for atom in gs]
    atoms.get_calculator().set_symbols(symbs)
    print(atoms.get_calculator().get_energy())
    exit()

    # Define parameters for equillibration
    equil_params = {
        "maxiter": 10 * N,
        "mode": "fixed"
    }

    nsteps = 200 * N
    calc = atoms.get_calculator()
    comp = {"Au": au_comp, "Cu": 1.0-au_comp}
    calc.set_composition(comp)
    energies = []
    for T in temperatures:
        mc = Montecarlo(atoms, T, accept_first_trial_move_after_reset=True)
        energy_obs = EnergyEvolution(mc)
        energies.append(energy_obs.energies)
        mc.attach(energy_obs, interval=100)
        mc.runMC(mode="fixed", steps=nsteps, equil_params=equil_params)
        thermo = mc.get_thermodynamic()
        thermo["converged"] = True
        thermo["temperature"] = T
        cf = calc.get_cf()
        db = dataset.connect("sqlite:///{}".format(db_name))
        tbl = db["results"]
        uid = tbl.insert(thermo)
        cf_tbl = db["corrfunc"]
        cf["runID"] = uid
        cf_tbl.insert(cf)

    fname = "data/atoms_{}_{}.xyz".format(atoms.get_chemical_formula(), thermo['energy'])
    write(fname, atoms)
    np.savetxt("data/energy_evolution_{}.csv".format(atoms.get_chemical_formula()),
                np.array(energies).T, delimiter=",")

def sa_fcc(au_comp):
    basis_elements = basis_elements = [['Cu', 'Au']]
    conc = Concentration(basis_elements=basis_elements)

    kwargs_fcc = {
        "crystalstructure": 'fcc',
        "a": 3.8,
        "size": [3, 3, 3],
        "concentration": conc,
        "db_name": 'cu-au_quad.db',
        "max_cluster_size": 4,
        "max_cluster_dia": 6.0,
        "cubic": False
        }
        
    with open("data/cluster_eci.json") as infile:
        eci = json.load(infile)
    print(eci)
    db_name = "data/cu-au3_lowstep.db"
    sa(au_comp, kwargs_fcc, eci, db_name, [10, 10, 10])


if __name__ == "__main__":
    comp = float(sys.argv[1])
    sa_fcc(comp)
