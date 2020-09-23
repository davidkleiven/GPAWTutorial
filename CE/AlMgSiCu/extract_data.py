from clease.settings import settings_from_json
from clease.data_manager import CorrFuncEnergyDataManager
from clease import ClusterCoverageChecker

db_name = "data/almgsicu_ce.db"
out = "data/almgsi.csv"

manager = CorrFuncEnergyDataManager(db_name, 'polynomial_cf')
manager.get_data([('converged', '=', 1)])
manager.to_csv(out)

# Generate coverage report
settings = settings_from_json("data/almgsicu_settings.json")
cov = ClusterCoverageChecker(settings, select_cond=[('converged', '=', True)])
cov.print_report()