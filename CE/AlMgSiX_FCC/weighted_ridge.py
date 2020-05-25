from clease import CorrFuncEnergyDataManager
from clease import GeneralizedRidgeRegression
import numpy as np
import json

db_name = '/home/gudrun/davidkl/Nedlastinger/MgSn.db'
dm = CorrFuncEnergyDataManager(db_name, 'binary_linear_cf')
X, y = dm.get_data([('struct_type', '=', 'initial')])
best_gcv = 100000.0
for _ in range(10000):
    alpha = np.random.lognormal(mean=-11, sigma=3.0, size=X.shape[1])
    #alpha = np.random.normal(scale=2.0, size=X.shape[1])
    ridge = GeneralizedRidgeRegression(alpha)
    coeff = ridge.fit(X, y)

    if ridge.opt_result['gcv'] < best_gcv:
        best_gcv = ridge.opt_result['gcv']
        print(f"New best GCV: {ridge.opt_result['gcv']}")
        data = {
            'alpha': list(ridge.alpha),
            'X': X.tolist(),
            'y': list(y),
            'gcv': ridge.opt_result['gcv'],
            'coeff': list(coeff),
            'names': dm._feat_names,
            'loo_dev': list(ridge.opt_result['gcv_dev']),
            'press_dev': list(ridge.opt_result['press_dev'])
        }

        with open("best_model.json", 'w') as out:
            json.dump(data, out)
