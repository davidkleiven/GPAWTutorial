import pandas as pd
from clease import CEBulk, Concentration, NewStructures, settingsFromJSON
from clease.tools import update_db
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
from clease.data_manager import (
    CorrelationFunctionGetterVolDepECI,
    CorrFuncVolumeDataManager
)
import traceback
from clease import PhysicalRidge
from clease.physical_ridge import random_cv_hyper_opt
import numpy as np
import re

db_local = "data/structures_with_bulk.db"
def initialize_db():
    db_name = "data/almgsiX_clease_voldep.db"

    settings = CEBulk(
        Concentration(basis_elements=[['Al', 'Mg', 'Si', 'X']]),
        crystalstructure='fcc',
        a=4.05, size=[1, 1, 1], max_cluster_size=3,
        max_cluster_dia=[5.0, 5.0],
        db_name=db_name
    )
    settings.basis_func_type = 'binary_linear'

    newStruct = NewStructures(settings)

    # Insert all initial structures
    counter = 0
    with connect(db_local) as db:
        for row in db.select():
            counter += 1
            print(f"Inserting structure {counter}")
            name = f"group{row.group}"
            atoms = row.toatoms()
            newStruct.insert_structure(atoms, name=name)
    
    data = pd.read_csv("data/bulk_mod_fit.csv")
    db = connect(settings.db_name)
    counter = 0
    for row in data.itertuples():
        print(f"Inserting final structure {counter}")
        counter += 1
        try:
            name = f"group{row[1]}"
            #print(name)
            #exit()
            E = row[2]
            B = row[3]
            V = row[4]
            dBdP = row[5]
            L = V**(1.0/3.0)
            atoms = Atoms(cell=[L, L, L])
            calc = SinglePointCalculator(atoms, energy=E)
            atoms.set_calculator(calc)

            print(name)
            init_id = db.get([('name', '=', name)]).id
            update_db(init_id, final_struct=atoms, db_name=settings.db_name,
                    custom_kvp_init={'bulk_mod': B, 'dBdP': dBdP})
        except Exception as exc:
            print(exc)
            traceback.print_exc()

    settings.save("data/settings_almgsiX_voldev.json")

#initialize_db()

def fit(fit_type="energy"):
    settings = settingsFromJSON("data/settings_almgsiX_voldev.json")
    cf_names = settings.all_cf_names

    if fit_type == 'energy':
        data = CorrelationFunctionGetterVolDepECI(
            "data/almgsiX_clease_voldep.db",
            "polynomial_cf",
            cf_names,
            order=5,
            properties=['energy', 'pressure', 'bulk_mod']   
        )

    X, y = data.get_data([('struct_type', '=', 'initial')])
    print(X[:, 0])

    regressor = PhysicalRidge(normalize=False)
    regressor.sizes = [int(n[1]) for n in data._feat_names]
    prog = re.compile(r"d(\d+)")
    regressor.diameters = []
    for cf_name in data._feat_names:
        result = prog.search(cf_name)
        if result is None:
            regressor.diameters.append(0.0)
        else:
            regressor.diameters.append(float(result.groups()[0]))
    params = {
            'lamb_dia': np.logspace(-12, 4, 5).tolist(),
            'lamb_size': np.logspace(-12, 4, 5).tolist(),
            'size_decay': ['linear', 'exponential'],
            'dia_decay': ['linear', 'exponential']
        }
    res = random_cv_hyper_opt(regressor, params, X, y, cv=5, num_trials=100)


#fit()
initialize_db()