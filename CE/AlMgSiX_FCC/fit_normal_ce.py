from clease.data_manager import CorrFuncEnergyDataManager
from clease import PhysicalRidge
from clease.physical_ridge import random_cv_hyper_opt
import numpy as np
import re
import json

db_name = "data/almgsiX_clease_voldep.db"
manager = CorrFuncEnergyDataManager(
    db_name, "binary_linear_cf"
)

X, y = manager.get_data([('struct_type', '=', 'initial')])
names = manager._feat_names
sizes = [int(n[1]) for n in names]
prog = re.compile(r"d(\d+)")
dia = []
for n in names:
    res = prog.findall(n)
    if not res:
        dia.append(0.0)
    else:
        dia.append(float(res[0]))

regressor = PhysicalRidge(normalize=False)
regressor.sizes = sizes
regressor.diameters = dia

params = {
            'lamb_dia': np.logspace(-6, 6, 5000).tolist(),
            'lamb_size': np.logspace(-6, 6, 5000).tolist(),
            'size_decay': ['linear', 'exponential', 'poly2', 'poly4', 'poly6'],
            'dia_decay': ['linear', 'exponential', 'poly2', 'poly4', 'poly6']
        }

res = random_cv_hyper_opt(regressor, params, X, y, cv=5, num_trials=10000)

outfile = "data/almgsix_normal_ce.json"
data = {
    'names': manager._feat_names,
    'coeff': res['best_coeffs'].tolist(),
    'X': X.tolist(),
    'y': y.tolist(),
    'cv': res['best_cv'],
    'eci': {n: c for n, c in zip(manager._feat_names, res['best_coeffs'])},
}

with open(outfile, 'w') as out:
    json.dump(data, out)
print(f"Results written to: {outfile}")