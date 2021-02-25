from ase.io.trajectory import TrajectoryWriter
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import niggli_reduce

#equiv_from_atat = [141, 155, 157, 158, 193, 68, 72, 75]
equiv_from_atat = []
ignore_formulas = ['Cu3', 'Al3', 'Mg3', 'Al4', 'Mg4', 'Cu4']
exclude_groups = [4001802753]
def main():
    db_name = "data/almgsicu.db"
    traj = TrajectoryWriter("data/almgsicu_data.traj")
    db = connect(db_name)
    groups = set()
    for row in db.select():
        g = row.get('group', None)
        if g is not None and g not in exclude_groups:
            groups.add(g)

    structures = []
    for g in groups:
        row = db.get(group=g, struct_type='initial')
        init = row.toatoms()
        if init.get_chemical_formula() in ignore_formulas:
            continue
        try:
            final = db.get(group=g, struct_type='relaxed')
        except:
            continue
        energy = 0.0
        if final is not None:
            energy = final.energy
            calc = SinglePointCalculator(init, energy=energy)
            init.calc = calc
            structures.append(init)
    
    for idx in sorted(equiv_from_atat, reverse=True):
        del structures[idx]
    for s in structures:
        traj.write(s)
        

main()