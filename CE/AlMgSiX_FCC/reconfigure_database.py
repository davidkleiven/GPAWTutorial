from clease import settingFromJSON, CorrFunction
from ase.db import connect
from clease.basis_function import BinaryLinear


setting = settingFromJSON("almgsixSettings3.json")
db = connect(setting.db_name)
db.delete_external_table('polynomial_cf')
db.delete_external_table('binary_linear_cf')
cf = CorrFunction(setting)
cf.reconfigure_db_entries()
