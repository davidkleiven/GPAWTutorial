from clease.settings import settings_from_json
from clease.data_manager import CorrFuncEnergyDataManager
from clease import ClusterCoverageChecker
import numpy as np

db_name = "data/cupd_ce.db"
out = "data/cupd.csv"

manager = CorrFuncEnergyDataManager(db_name, 'polynomial_cf')
manager.get_data([('converged', '=', 1)])

duplicates = []
X = manager._X
for i in range(X.shape[0]):
    new_duplicates = set()
    for j in range(i+1, X.shape[0]):
        if np.allclose(X[i, :], X[j, :]):
            new_duplicates.add(i)
            new_duplicates.add(j)
    if new_duplicates:
        duplicates.append(new_duplicates)
print(duplicates)
manager.to_csv(out)

# Generate coverage report
settings = settings_from_json("data/cupd_settings.json")
cov = ClusterCoverageChecker(settings, select_cond=[('converged', '=', True)])
cov.print_report()