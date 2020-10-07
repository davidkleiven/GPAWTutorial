from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from clease.tools import update_db
from clease import NewStructures, CEBulk, Concentration
from itertools import product
from ase.build import bulk
from ase.eos import EquationOfState
from ase.db import connect
import numpy as np

a_cu = 3.597
a_au = 4.065

def settings():
    return CEBulk(
        Concentration(basis_elements=[['Au', 'Cu']]),
        a=4.0,
        crystalstructure='fcc',
        max_cluster_size=4,
        max_cluster_dia=[12.0, 5.0, 5.0],
        db_name="data/aucu_emt_demo.db",
        size=[1, 1, 1]
    )

def reconfigure():
    cebulk = settings()
    print(cebulk.multiplicity_factor)
    #exit()
    from clease import CorrFunction
    cf = CorrFunction(cebulk)
    #cf.reconfigure_db_entries()


def populate_db():
    cebulk = settings()
    new_struct = NewStructures(cebulk)
    atoms = bulk('Au', a=4.0, cubic=True)*(1, 2, 2)

    for symb in product(['Au', 'Cu'], repeat=len(atoms)):
        atoms_cpy = atoms.copy()
        atoms_cpy.symbols = symb
        new_struct.insert_structure(atoms_cpy)

def calculate():
    db = connect("data/aucu_emt_demo.db")
    for row in db.select([('struct_type', '=', 'initial')]):
        print(f"Calculating {row.id}")
        atoms = row.toatoms()
        cu_conc = np.count_nonzero(atoms.numbers == 29)/len(atoms)
        a = a_cu*cu_conc + (1-cu_conc)*a_au
        cell = atoms.get_cell()
        v = np.linalg.det(cell)/len(atoms)
        target = a**3/4
        ratio = (target/v)**(1.0/3.0)
        atoms.set_cell(cell*ratio, scale_atoms=True)
        cell = atoms.get_cell()

        calc = EMT()
        atoms.set_calculator(calc)
        volumes = []
        energies = []
        for factor in np.linspace(0.95, 1.2, 10):
            atoms.set_cell(cell*factor, scale_atoms=True)
            volumes.append(atoms.get_volume())
            energies.append(atoms.get_potential_energy())
        eos = EquationOfState(volumes, energies, eos='murnaghan')
        eos.fit()
        e0, B, dBdP, v0 = eos.eos_parameters
        scale = v0/np.linalg.det(cell)
        atoms.set_cell(cell*scale**(1.0/3.0), scale_atoms=True)
        single = SinglePointCalculator(atoms, energy=e0)
        atoms.set_calculator(single)
        update_db(uid_initial=row.id, final_struct=atoms,
                  db_name="data/aucu_emt_demo.db", 
                  custom_kvp_init={'bulk_mod': B, 'dBdP': dBdP})

#populate_db()
#calculate()
reconfigure()
settings().save("data/aucu_emt_demo_settings.json")

