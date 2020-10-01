from ase.calculators.emt import EMT
from ase.db import connect
from ase.constraints import UnitCellFilter
from ase.optimize.precon import PreconLBFGS
from ase.build import niggli_reduce

DB_NAME = "data/cupd.db"

def main(runId):
    db = connect(DB_NAME)
    row = db.get(id=runId)
    group = row.group
    atoms = row.toatoms()
    niggli_reduce(atoms)
    calc = EMT()
    atoms.set_calculator(calc)
    ucell = UnitCellFilter(atoms)
    opt = PreconLBFGS(ucell)
    opt.run(fmax=0.02, smax=0.003)

    db.write(atoms, group=group, struct_type='relaxed')

for i in range(151, 691):
    print(f"RUNNING {i}")
    main(i)