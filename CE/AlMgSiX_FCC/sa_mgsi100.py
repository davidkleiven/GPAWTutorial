from clease.settings import settings_from_json
import json
from clease.montecarlo import Montecarlo
from clease.calculator import attach_calculator
from ase.build import bulk
from ase.io import write


def test_mgsi100():
    eci = {}
    with open("data/almgsix_normal_ce.json", 'r') as infile:
        data = json.load(infile)
        eci = data['eci']

    settings = settings_from_json("data/settings_almgsiX_voldev.json")
    settings.basis_func_type = "binary_linear"

    atoms = bulk('Al', a=4.05, cubic=True)*(4, 4, 4)
    atoms = attach_calculator(settings, atoms, eci)
    for i in range(int(len(atoms)/2)):
        atoms[i].symbol = 'Mg'
        atoms[i+int(len(atoms)/2)].symbol = 'Si'
  
    mc = Montecarlo(atoms, 1000)
    temps = [1000, 800, 600, 500, 400, 300, 200, 100]
    for T in temps:
        mc.T = T
        mc.run(steps=100*len(atoms))

    fname = "data/" + atoms.get_chemical_formula() + "_mc_test.xyz"
    write(fname, atoms)

test_mgsi100()