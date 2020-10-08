from clease import CorrFuncEnergyDataManager

manager = CorrFuncEnergyDataManager("almgsi_new.db", "polynomial_cf")
manager.get_data([("converged", "=", 1)])
manager.to_csv("data/almgsi.csv")
