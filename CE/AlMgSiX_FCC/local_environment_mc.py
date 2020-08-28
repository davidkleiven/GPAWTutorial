from clease.montecarlo.constraints import MCConstraint
import numpy as np
import dataset
from ase.io import read
from clease.montecarlo import SGCMonteCarlo
from clease.montecarlo.observers import MCObserver, Snapshot
from clease.tools import species_chempot2eci
import sys
import random
import json
from clease import settingsFromJSON
from clease.calculator import attach_calculator


class ActiveElementConstraint(MCConstraint):
    def __init__(self, atoms, active_sites):
        self.can_move = np.zeros(len(atoms), dtype=np.uint8)
        self.can_move[active_sites] = 1

    def __call__(self, system_changes):
        for change in system_changes:
            if not self.can_move[change[0]]:
                return False
        return True

class TaggedSiteSymbol(MCObserver):
    def __init__(self, symbol, atoms):
        self.tags = [atom.tag for atom in atoms]
        self.symbol = symbol
        self.occurences = {k: 0 for k in set(self.tags) if k > -1}
        self.current_number = {k: 0 for k in self.occurences.keys()}
        self.current_number_sq = {k: 0 for k in self.occurences.keys()}
        self.num_calls = 0

    def __call__(self, system_changes):
        for change in system_changes:
            idx = change[0]
            if change[1] == self.symbol:
                self.occurences[self.tags[idx]] -= 1
            if change[2] == self.symbol:
                self.occurences[self.tags[idx]] += 1
        for k, v in self.occurences.items():
            self.current_number[k] += v
            self.current_number_sq[k] += v**2
        self.num_calls += 1

    def averages(self):
        #return self.occurences
        avg = {k: v/self.num_calls for k, v in self.current_number.items()}
        return avg
    
    def reset(self):
        self.num_calls = 0
        self.current_number = {k: 0 for k in self.current_number.keys()}
    

def main(argv):
    fname = argv[0]
    chem_pot = argv[1:]
    initial = read(fname)
    db = dataset.connect("sqlite:///data/almgsi_mc_sgc.db")
    tbl = db['local_environ_mgsi']
    runID = hex(random.randint(0, 2**32-1))

    eci = {}
    with open("data/almgsix_normal_ce.json", 'r') as infile:
        data = json.load(infile)
        eci = data['eci']

    cf_names = [k for k in eci.keys() if k[1] == '2']
    settings = settingsFromJSON("data/settings_almgsiX_voldev.json")
    settings.basis_func_type = "binary_linear"
    atoms = attach_calculator(settings, initial.copy(), eci)

    atoms.numbers = initial.numbers
    for i, atom in enumerate(initial):
        atoms[i].tag = atom.tag

    mc = SGCMonteCarlo(atoms, 1000, symbols=['Al', 'X'])

    temps = [1000, 800, 700, 600, 500, 400, 300, 200, 100]

    active_sites = [i for i, atom in enumerate(atoms) if atom.tag != -1]
    constraint = ActiveElementConstraint(atoms, active_sites)
    observer = TaggedSiteSymbol('X', atoms)
    mc.add_constraint(constraint)
    mc.attach(observer)

    snap_fname = f"/work/sophus/local_environ_mc/traj/run{runID}"
    snap = Snapshot(fname=snap_fname, atoms=atoms)
    mc.attach(snap, interval=10*len(atoms))

    chem_pot_dict = {
        'Mg': 0.0,
        'Si': 0.0,
        'X': float(chem_pot[0])
    }
    chem_pot_dict = species_chempot2eci(settings.basis_functions, chem_pot_dict)
    for T in temps:
        mc.T = T
        mc.run(steps=100*len(atoms), chem_pot=chem_pot_dict)
        thermo = mc.get_thermodynamic_quantities()
        thermo['runID'] = runID
        thermo['initial'] = fname
        avgOcupp = {f'occupLayer{k}': v for k, v in observer.averages().items()}
        thermo.update(avgOcupp)
        tbl.insert(thermo)

main(sys.argv[1:])
