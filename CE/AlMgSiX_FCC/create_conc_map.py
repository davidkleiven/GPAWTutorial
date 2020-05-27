from itertools import product
import numpy as np

cmd = "nice -19 python3 almgsix_sgc.py"
concs = [0.0, 0.1, 0.2, 0.25, 0.375, 0.5, 0.675, 0.75, 0.8, 0.9]

fname = "data/conc_array.txt"

with open(fname, 'w') as out:
    for conc in product(concs, repeat=3):
        if np.any(np.array(conc) > 0.01) and np.all(np.array(conc) < 0.99) and sum(conc) < 1.01:
            out.write(f"{cmd} {conc[0]} {conc[1]} {conc[2]}\n")