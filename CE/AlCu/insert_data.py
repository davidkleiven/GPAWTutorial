from clease.settings import settings_from_json
from clease import NewStructures
from ase.db import connect

settings = settings_from_json("data/cupd_settings.json")
dft_db = "data/cupd.db"

def main():
    initial_structs = []
    final_structs = []
    with connect(dft_db) as db:
        groups = set()
        for row in db.select():
            groups.add(row.get('group', -1))

        for g in groups:
            try:
                init = db.get([('group', '=', g), ('struct_type', '=', 'initial')])
                final = db.get([('group', '=', g), ('struct_type', '=', 'relaxed')])
                initial_structs.append(init.toatoms())
                final_structs.append(final.toatoms())
            except KeyError:
                pass

    # Sanity checks
    for i, f in zip(initial_structs, final_structs):
        i_formula = i.get_chemical_formula()
        f_formula = f.get_chemical_formula()
        assert i_formula == f_formula

    print(f"Extracted {len(initial_structs)} structures")

    new_struct = NewStructures(settings)
    for i, f in zip(initial_structs, final_structs):
        print(f"Inserting struct {i.get_chemical_formula()}")
        try:
            new_struct.insert_structure(i, f)
        except:
            pass

main()
