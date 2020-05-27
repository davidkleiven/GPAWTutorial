from clease.montecarlo import SGCMonteCarlo, Montecarlo
from clease.montecarlo.observers import CorrelationFunctionObserver
from clease import settingsFromJSON
from clease.calculator import attach_calculator
from clease.tools import species_chempot2eci
import json
from ase.build import bulk
import sys
import dataset
import random
import numpy as np
from ase.io import write


def main(argv):
    db = dataset.connect("sqlite:///data/almgsi_mc_sgc.db")
    tbl = db['random_direction_sa']
    runID = hex(random.randint(0, 2**32-1))

    eci = {}
    with open("data/almgsix_normal_ce.json", 'r') as infile:
        data = json.load(infile)
        eci = data['eci']

    cf_names = [k for k in eci.keys() if k[1] == '2']
    settings = settingsFromJSON("data/settings_almgsiX_voldev.json")
    settings.basis_func_type = "binary_linear"
    #print(settings.basis_functions)
    #exit()
    atoms = bulk('Al', a=4.05, cubic=True)*(4, 4, 4)
    atoms = attach_calculator(settings, atoms, eci)

    obs = CorrelationFunctionObserver(atoms.get_calculator(), names=cf_names)

    temps = [1000, 800, 600, 500, 400, 300, 200]
    chem_pot = {
        'Mg': float(argv[0]),
        'Si': float(argv[1]),
        'X': float(argv[2]),
    }
    chem_pot = species_chempot2eci(settings.basis_functions, chem_pot)

    mc = SGCMonteCarlo(atoms, 1000, symbols=['Al', 'Mg', 'Si', 'X'])
    mc.attach(obs, interval=100)
    for T in temps:
        mc.T = T
        mc.run(steps=100*len(atoms), chem_pot=chem_pot)
        thermo = mc.get_thermodynamic_quantities()
        cfs = obs.get_averages()
        cfs = {k: v for k, v in cfs.items() if 'c2' in k}
        mc.reset()
        thermo.update(cfs)
        thermo['runID'] = runID
        tbl.insert(thermo)

def randomDirectionSearch(num_attempts):
    min_value = -4.0
    max_value = 10.0
    center = 0.5*(min_value + max_value)

    ds = 0.5
    for i in range(num_attempts):
        start = np.random.rand(3)*(max_value - min_value) + min_value
        direction = np.array([np.random.normal() for i in range(3)])
        direction /= np.sqrt(np.sum(direction**2))
        counter = 1
        while True:
            chem_pot = start + ds*counter*direction
            print(f"Counter {counter}: {chem_pot}")
            if np.any(chem_pot > max_value) or np.any(chem_pot < min_value):
                break
            main(chem_pot)
            counter += 1

def almg_rich(num_attempts):
    for i in range(num_attempts):
        chem_pot = [
            np.random.normal(loc=2.0, scale=2.0),
            np.random.normal(loc=-10.0, scale=2.0),
            np.random.normal(loc=0.0, scale=2.0)
        ]
        print(f"Run no {i}")
        main(chem_pot)

def fixed_comp(conc_array):
    db = dataset.connect("sqlite:///data/almgsi_mc_sgc.db")
    tbl = db['sa_fixed_conc']
    runID = hex(random.randint(0, 2**32-1))

    conc = {
        'Mg': float(conc_array[0]),
        'Si': float(conc_array[1]),
        'X': float(conc_array[2])
    }

    eci = {}
    with open("data/almgsix_normal_ce.json", 'r') as infile:
        data = json.load(infile)
        eci = data['eci']

    settings = settingsFromJSON("data/settings_almgsiX_voldev.json")
    settings.basis_func_type = "binary_linear"
    #print(settings.basis_functions)
    #exit()
    atoms = bulk('Al', a=4.05, cubic=True)*(4, 4, 4)
    atoms = attach_calculator(settings, atoms, eci)
    start = 0
    for k, v in conc.items():
        num = int(v*len(atoms))
        for i in range(start, start+num):
            atoms[i].symbol = k
        start += num
    
    mc = Montecarlo(atoms, 1000)
    temps = [1000, 800, 600, 500, 400, 300, 200, 100]
    for T in temps:
        mc.T = T
        mc.run(steps=10*len(atoms))
        thermo = mc.get_thermodynamic_quantities()
        thermo['runID'] = runID
        tbl.insert(thermo)

    fname = "data/gs/" + atoms.get_chemical_formula() + str(runID) + ".xyz"
    write(fname, atoms)

#randomDirectionSearch(10000)
#almg_rich(1000)
fixed_comp(sys.argv[1:])
