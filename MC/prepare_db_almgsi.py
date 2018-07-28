import dataset
import numpy as np

db = dataset.connect("sqlite:///data/enthalpy_formation_almgsi.db")
mg_concs = np.linspace(0.0,1.0,21)
si_concs = np.linspace(0.0,0.5,11)

tbl = db["systems"]

for c_mg in mg_concs:
    for c_si in si_concs:
        row = dict(mg_conc=c_mg,si_conc=c_si,status="new")
        tbl.insert(row)
