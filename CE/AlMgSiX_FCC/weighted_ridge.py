from clease import CorrFuncEnergyDataManager
from clease import GeneralizedRidgeRegression
import numpy as np
import json

db_name = 'data/almgsiX_clease_voldep.db'
db_name = '/home/davidkleiven/Downloads/MgSn.db'
dm = CorrFuncEnergyDataManager(db_name, 'binary_linear_cf')
X, y = dm.get_data([('struct_type', '=', 'initial')])
best_gcv = 100000.0
for _ in range(10000):
    alpha = np.random.lognormal(mean=-10, sigma=2.0, size=X.shape[1])
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
            'names': dm._feat_names
        }

        with open("best_model.json", 'w') as out:
            json.dump(data, out)

pred = X.dot(coeff)
print(np.sqrt(np.mean((y - pred)**2)))
#print(ridge.alpha)
