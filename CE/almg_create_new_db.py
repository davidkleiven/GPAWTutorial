from ase.clease import CEBulk as BulkCrystal
from ase.clease import NewStructures as GenerateStructures
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator

def main():
    db_name_in = "ce_hydrostatic_new_config.db"
    db_name_out = "ce_hydrostatic_only_relaxed.db"
    conc_args = {
        "conc_ratio_min_1":[[1,0]],
        "conc_ratio_max_1":[[0,1]],
    }
    ceBulk = BulkCrystal( crystalstructure="fcc", a=4.05, size=[4,4,4], \
                            basis_elements=[["Al","Mg"]], conc_args=conc_args,
                            db_name=db_name_out,
                            max_cluster_size=4)

    struct_gen = GenerateStructures( ceBulk )
    db_in = connect(db_name_in)
    for row in db_in.select(converged=1):
        temp_atoms = ceBulk.atoms.copy()
        print(row.energy)
        atoms = row.toatoms()
        for i, atom in enumerate(atoms):
            temp_atoms[i].symbol = atom.symbol

        final = temp_atoms.copy()
        calc = SinglePointCalculator(final, energy=row.energy)
        final.set_calculator(calc)
        struct_gen.insert_structure(init_struct=temp_atoms, final_struct=final)

if __name__ == "__main__":
    main()
