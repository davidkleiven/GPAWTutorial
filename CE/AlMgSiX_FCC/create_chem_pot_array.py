"""
Creates a simulation array to be used together with the script almgsix_sgc.py
"""
import numpy as np
from itertools import product
outfile = "data/chem_pot_array.txt"
cmd = "nice -19 python3 almgsix_sgc.py"
potentials = np.linspace(-6.0, 0.0, 4)

with open(outfile, 'w') as out:
    for comb in product(range(len(potentials)), repeat=3):
        out.write(f"{cmd} {potentials[comb[0]]} {potentials[comb[1]]} {potentials[comb[2]]}\n")

