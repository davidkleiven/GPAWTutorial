from clease import settingFromJSON
from clease.montecarlo.constraints import PairConstraint
from clease.montecarlo import Montecarlo
from clease.calculator import attach_calculator
import json
from random import choice
from ase.io.trajectory import TrajectoryWriter
from ase.build import bulk

FNAME = "random_settings.json"
ECI_FILE = "data/almgsiX_eci.json"

def get_gs():
    atoms = bulk('Al', cubic=True)*(5, 1, 1)
    setting = settingFromJSON(FNAME)

    with open(ECI_FILE, 'r') as infile:
        eci = json.load(infile)
    
    atoms = attach_calculator(setting=setting, atoms=atoms, eci=eci)
    setting.set_active_template(atoms=atoms)
    cluster = setting.cluster_list.get_by_name("c2_d0000_0")[0]

    cnst = PairConstraint(['X', 'X'], cluster, setting.trans_matrix, atoms)

    symbols = ['Al', 'Mg', 'Si', 'X']
    numX = 0
    removed = False
    for atom in atoms:
        new_symb = choice(symbols)
        atom.symbol = new_symb

        if new_symb == 'X':
            numX += 1
        
        if numX == 1 and not removed:
            symbols.remove('X')
            removed = True

    T = [1000, 800, 600, 400, 200, 100, 50]
    for temp in T:
        mc = Montecarlo(atoms, temp)
        mc.add_constraint(cnst)
        mc.run(steps=10*len(atoms))

    return atoms

def generate_restricted_gs():
    traj = TrajectoryWriter("data/restricted_gs_5x1x1.traj", mode='a')
    N = 400
    for i in range(N):
        print("{} of {}".format(i, N))
        atoms = get_gs()
        traj.write(atoms)

generate_restricted_gs()


