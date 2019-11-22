from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms

db_name = 'almgsiX_dft.db'

def port_almgsi_db():
    db_new = connect(db_name)

    names = []
    for row in db.select(converged=1):
        names.append(row.name)

    group = 0
    for name in names:
        ignore = False
        print(name)
        init_struct = None
        final_struct = None
        energy = 0.0
        num_occ = sum(1 for _ in db.select(name=name))
        for row in db.select(name=name):
            if row["calculator"] == "unknown":
                energy = row.energy
                init_struct = row.toatoms()

                if num_occ != 2:
                    final_struct = row.toatoms()
            elif row["calculator"] == 'none':
                init_struct = row.toatoms()

                fid = row.get('final_struct_id', -1)
                if fid != -1:
                    final_struct = db.get(id=fid)
                    energy = db.get(id=fid).get('energy', None)
                    if energy is None:
                        ignore = True
                    break
            else:
                final_struct = row.toatoms()

        if isinstance(final_struct, Atoms):
            calc = SinglePointCalculator(final_struct, energy=energy)
            final_struct.set_calculator(calc)
        db_new.write(init_struct, group=group, type="initial")
        db_new.write(final_struct, group=group, type="full_relax")
        group += 1

def transfer_from_vac_db():
    db_new = connect(db_name)
    db = connect('almgsiX_fcc.db')
    names = []
    for row in db.select(converged=1):
        names.append(row.name)

    group = 288
    for name in names: 
        init = db.get(name=name, struct_type='initial')
        atoms = init.toatoms()
        numX = sum(1 for atom in atoms if atom.symbol == 'X')
        if numX == 0:
            continue
        final_id = init.final_struct_id
        final = db.get(id=final_id)
        db_new.write(atoms, group=group, type='initial')
        db_new.write(final, group=group, type='full_relax')
        group += 1
#port_almgsi_db()
#transfer_from_vac_db()