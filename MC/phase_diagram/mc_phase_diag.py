import sys
from ase.clease import Concentration
from ase.clease import CEBulk as BulkCrystal
import json
from cemc import get_ce_calc
from cemc.tools import GSFinder
import numpy as np
import dataset

DB_NAME = "data/almgsi_10x10x10.db"
ECI_FILE = "data/almgsi_ga_eci.json"
PHASE_DIAG_DB = "sqlite:////work/sophus/almgsi_phasediag/phase_diag.db"

def get_ce_with_calc():
    conc = Concentration(basis_elements=[["Al","Mg","Si"]])
    kwargs = {
        "crystalstructure": "fcc",
        "a": 4.05,
        "size": [4, 4, 4],
        "concentration": conc,
        "db_name": DB_NAME,
        "max_cluster_size": 4
    }
    ceBulk = BulkCrystal(**kwargs)
    
    with open(ECI_FILE, 'r') as infile:
        ecis = json.load(infile)

    db_name = "large_cell_db10x10x10.db"
    calc = get_ce_calc(ceBulk, kwargs, ecis, size=[10, 10, 10], db_name=db_name)
    ceBulk = calc.BC
    ceBulk.atoms.set_calculator(calc)
    return ceBulk
    
def find_gs(formula="MgSi"):
    ceBulk = get_ce_with_calc()
    symbs = ["Mg" for _ in range(len(ceBulk.atoms))]

    if formula == "MgSi":
        for i in range(int(len(symbs)/2), len(symbs)):
            symbs[i] = "Si"
    elif formula == "Mg3Si":
        for i in range(int(3*len(symbs)/4), len(symbs)):
            symbs[i] = "Si"
    ceBulk.atoms.get_calculator().set_symbols(symbs)
    gs = GSFinder()
    temps = np.linspace(1.0, 1500, 30)[::-1]
    result = gs.get_gs(ceBulk, temps=temps, n_steps_per_temp=10000)
    fname = "data/ground_state{}.xyz".format(formula)
    
    from ase.io import write
    write(fname, result["atoms"])
    print("Atoms written to {}".format(fname))

def generate_phase_diag_plan():
    fname = "data/phase_diagram_plan.json"
    db = dataset.connect(PHASE_DIAG_DB)
    tbl = db["simulation_plan"]
    tbl.insert({
            "num_insert": 0,
            "phase": "mgsi",
            "swap_old": "Mg",
            "swap_new": "Al"
        })

    for num in range(2, 20, 2):
        entry = {
            "num_insert": num,
            "phase": "mgsi",
            "swap_old": "Mg",
            "swap_new": "Al"
        }
        tbl.insert(entry)
        entry = {
            "num_insert": num,
            "phase": "mgsi",
            "swap_old": "Si",
            "swap_new": "Al"
        }
        tbl.insert(entry)
        entry = {
            "num_insert": num,
            "phase": "mgsi",
            "swap_old": "Si",
            "swap_new": "Mg"
        }
        tbl.insert(entry)
        entry = {
            "num_insert": num,
            "phase": "Al",
            "swap_old": "Al",
            "swap_new": "Mg"
        }
        tbl.insert(entry)

        entry = {
            "num_insert": num,
            "phase": "Al",
            "swap_old": "Al",
            "swap_new": "Si"
        }
        tbl.insert(entry)

        entry = {
            "num_insert": num,
            "phase": "Al",
            "swap_old": "Al",
            "swap_new": "Mg-Si"
        }
        tbl.insert(entry)

def perform_mc(uid):
    from ase.io import read
    from cemc.mcmc import Montecarlo
    db = dataset.connect(PHASE_DIAG_DB)
    tbl = db["simulation_plan"]
    row = tbl.find_one(id=uid)
    print(row)
    temperatures = list(np.arange(1, 600, 50))

    ceBulk = get_ce_with_calc()
    swap_symbs = row["swap_new"].split("-")
    current = 0
    if row["phase"] == "mgsi":
        atoms = read("data/ground_stateMgSi.xyz")
        symbols = [atom.symbol for atom in atoms]
    else:
        symbols = ["Al" for _ in range(len(ceBulk.atoms))]

    num_inserted = np.zeros(len(swap_symbs))

    for i in range(len(symbols)):
        if symbols[i] == row["swap_old"]:
            symbols[i] = swap_symbs[current]
            num_inserted[current] += 1
        if num_inserted[-1] >= row["num_insert"]:
            break
        elif num_inserted[current] >= row["num_insert"]:
            current += 1
        elif num_inserted[current] >= row["num_insert"]:
            current += 1

    ceBulk.atoms.get_calculator().set_symbols(symbols)
    result_tab = db["simulations"]
    for T in temperatures:
        print("Current temperature: {}".format(T))
        mc = Montecarlo(ceBulk.atoms, T)
        equil_params = {
            "mode": "fixed",
            "maxiter": 10000
        }
        mc.runMC(steps=1000*len(symbols), equil_params=equil_params)
        thermo = mc.get_thermodynamic()
        thermo["runID"] = uid
        result_tab.insert(thermo)

if __name__ == "__main__":
    option = "gs"
    formula = "MgSi"
    uid = 0
    for arg in sys.argv:
        if "--option=" in arg:
            option = arg.split("--option=")[-1]
        elif "--formula=" in arg:
            formula = arg.split("--formula=")[-1]
        elif "--uid=" in arg:
            uid = int(arg.split("--uid=")[1])
    
    if option == "gs":
        find_gs(formula=formula)
    elif option == "plan":
        generate_phase_diag_plan()
    elif option == "run":
        print("Running UID {}".format(uid))
        perform_mc(uid)
        

