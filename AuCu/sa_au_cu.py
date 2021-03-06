import dataset
from cemc.mcmc import Montecarlo
from cemc import get_atoms_with_ce_calc
import sys
import json
from ase.ce import CEBulk, Concentration
from ase.io import write


def sa(au_comp, kwargs, eci, db_name, size):
    bc = CEBulk(**kwargs)
    atoms = get_atoms_with_ce_calc(bc, kwargs, eci, size)
    temperatures = [800, 700, 600, 500, 400, 300, 200, 100, 50, 25, 10]
    N = len(bc.atoms)

    # Define parameters for equillibration
    equil_params = {
        "maxiter": 10 * N,
        "mode": "fixed"
    }

    comps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    nsteps = 100 * N
    calc = atoms.get_calculator()
    for au_comp in comps:
        comp = {
            "Au": au_comp,
            "Cu": 1.0-au_comp
        }
        calc.set_composition(comp)
        for T in temperatures:
            mc = Montecarlo(atoms, T)
            mc.runMC(mode="fixed", steps=nsteps, equil_params=equil_params)
            thermo = mc.get_thermodynamic()
            thermo["converged"] = True
            thermo["au_conc"] = au_comp
            thermo["temperature"] = T
            cf = calc.get_cf()
            db = dataset.connect("sqlite:///{}".format(db_name))
            tbl = db["results"]
            uid = tbl.insert(thermo)
            cf_tbl = db["corrfunc"]
            cf["runID"] = uid
            cf_tbl.insert(cf)

            fname = "data/atoms_{}.xyz".format(atoms.get_chemical_formula())
            write(fname, atoms)


def sa_fcc(al_comp):
    alat = 3.9 # Something in between Cu and Au
    alat = 3.8
    kwargs_fcc = {
        "crystalstructure": 'fcc',
        "a": 3.8,
        "size": [4, 4, 2],
        "concentration": Concentration(basis_elements=[["Au", "Cu"]]),
        "db_name": 'cu-au.db',
        "max_cluster_size": 2,
        "max_cluster_dist": 1.5*alat
        }
        
    with open("data/eci_aucu.json") as infile:
        eci = json.load(infile)
    db_name = "data/sa_aucu_only_pairs.db"
    sa(al_comp, kwargs_fcc, eci, db_name, [10, 10, 10])


if __name__ == "__main__":
    comp = float(sys.argv[1])
    sa_fcc(comp)
