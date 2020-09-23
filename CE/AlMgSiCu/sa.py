import sys
from ase.db import connect
from clease.montecarlo import Montecarlo
from clease.montecarlo.observers import LowestEnergyStructure
from clease.settings import settings_from_json
from clease.calculator import attach_calculator
from ase.build import bulk
from random import shuffle
from ase.calculators.singlepoint import SinglePointCalculator
import json

N = 4
eci_file = "data/ga_almgsicu_model.json"
db_name = "data/almgsicu_sa.db"
conc_file = "data/sa_concs.csv"
settings_file = "data/almgsicu_settings.json"

def get_conc(rowNum):
    with open(conc_file, 'r') as infile:
        lines = [line.strip() for line in infile]
    if rowNum > len(lines):
        return None
    symbs = ['Al', 'Mg', 'Si', 'Cu']
    concs = {s: float(c) for s, c in zip(symbs, lines[rowNum].split(','))}
    s = sum(concs.values())
    return {k: v/s for k, v in concs.items()}

def get_eci():
    with open(eci_file, 'r') as infile:
        data = json.load(infile)
    return data['Coeffs']

def main(rowNum):
    conc = get_conc(rowNum)
    if conc is None:
        return

    settings = settings_from_json(settings_file)
    atoms = bulk('Al', a=4.05, crystalstructure='fcc', cubic=True)*(N, N, N)

    atoms = attach_calculator(settings, atoms, get_eci())
    cu = ['Cu']*int(conc['Cu']*len(atoms))
    si = ['Si']*int(conc['Si']*len(atoms))
    mg = ['Mg']*int(conc['Mg']*len(atoms))
    al = ['Al']*(len(atoms) - len(cu) - len(mg) - len(si))
    symbols = al+mg+si+cu
    shuffle(symbols)
    atoms.symbols = symbols

    # Trigger a calculation
    atoms.get_potential_energy()
    temperatures = list(range(1000, 1, -50))
    mc = Montecarlo(atoms, 60)
    obs = LowestEnergyStructure(atoms)
    mc.attach(obs)

    for T in temperatures:
        mc.T = T
        mc.run(steps=20*len(atoms))
    
    db = connect(db_name)
    obs.emin_atoms.set_calculator(SinglePointCalculator(energy=obs.lowest_energy))
    db.write(obs.emin_atoms)

main(int(sys.argv[1]))
