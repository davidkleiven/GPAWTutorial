import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'font.family': 'serif', 'font.size': 11, 'svg.fonttype': 'none'})
from matplotlib import pyplot as plt
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
import json
import sys
from ase.units import GPa

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
    cf_names = [n for n in cf_names if int(n[1]) < 3]

    if fit_type == "energy":
        data = CorrelationFunctionGetterVolDepECI(
            "data/almgsiX_clease_voldep.db",
            "binary_linear_cf",
            cf_names,
            order=10,
            properties=['energy', 'pressure', 'bulk_mod', 'dBdP']   
        )
    elif fit_type == "volume":
        data = CorrFuncVolumeDataManager(
             "data/almgsiX_clease_voldep.db",
            "binary_linear_cf",
            cf_names
        )

    skip_groups = [5, 13, 42, 71, 84, 1490300799, 3797755424]
    scond = [('struct_type', '=', 'initial')]
    for sg in skip_groups:
        scond.append(('name', '!=', f'group{sg}'))
    X, y = data.get_data(scond)
    N = X.shape[0]

    #X[int(N/3):int(2*N/3)] *= 1000.0
    #y[int(N/3):] *= 1000.0

    regressor = PhysicalRidge(normalize=False, reuse_svd=True)
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
            'lamb_dia': np.logspace(-12, 4, 5000).tolist(),
            'lamb_size': np.logspace(-12, 4, 5000).tolist(),
            'size_decay': ['linear', 'exponential', 'poly2', 'poly4', 'poly6'],
            'dia_decay': ['linear', 'exponential', 'poly2', 'poly4', 'poly6']
        }
    res = random_cv_hyper_opt(regressor, params, X, y, cv=5, num_trials=5000)

    outfile = f"data/voldep_fit_result{fit_type}.json"
    data = {
        'names': cf_names,
        'coeff': res['best_coeffs'].tolist(),
        'X': X.tolist(),
        'y': y.tolist(),
        'cv': res['best_cv']
    }

    with open(outfile, 'w') as out:
        json.dump(data, out)
    print(f"Results written to: {outfile}")

def plotvol():
    with open(f"data/voldep_fit_resultvolume.json", 'r') as infile:
        data = json.load(infile)

    X = np.array(data['X'])
    y = np.array(data['y'])
    coeff = data['coeff']
    pred = X.dot(coeff)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot()
    xmin = np.min(y)
    xmax = np.max(y)
    rng = xmax - xmin
    if rng < 1e-3:
        rng = 1e-3
    xmin -= 0.01*rng
    xmax += 0.01*rng
    ax.plot([xmin, xmax], [xmin, xmax], color='#1c1c14')
    
    ax.plot(y, pred, 'o', mfc='none', color='#742d18', markersize=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("CE volume")
    ax.set_ylabel("DFT volume")
    fig.tight_layout()
    fig.savefig("fig/volume_fit.pdf")
    fig.savefig("fig/volume_fit.png")

    
def plot():
    with open(f"data/voldep_fit_resultenergy.json", 'r') as infile:
        data = json.load(infile)
    X = np.array(data['X'])
    y = np.array(data['y'])
    coeff = data['coeff']

    plots_info = [
        {
            'ylabel': 'Energy CE (eV/atom)',
            'xlabel': 'Energy DFT (eV/atom)',
            'fname': 'fig/energy_pred.pdf',
            'factor': 1.0,
        },
        {
            'ylabel': 'Pressure CE (GPa)',
            'xlabel': 'Pressure DFT (GPa)',
            'fname': 'fig/pressure_pred.pdf',
            'factor': 1.0/GPa
        },
        {
            'ylabel': 'Bulk mod. CE (GPa)',
            'xlabel': 'Bulk mod. DFT (GPa)',
            'fname': 'fig/bulk_mod.pdf',
            'factor': 1.0/GPa
        },
        {
            'ylabel': r'$dB/dP$ (CE)',
            'xlabel': r'$dB/dP$ (DFT)',
            'fname': 'fig/dBdP.pdf',
            'factor': 1.0
        }
    ]

    pred = X.dot(coeff)
    for i, info in enumerate(plots_info):
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(1, 1, 1)
        start = int(i*len(y)/4)
        end = int((i+1)*len(y)/4)
        dft_data = y[start:end]*info['factor']
        ce_data = pred[start:end]*info['factor']

        xmin = np.min(dft_data)
        xmax = np.max(dft_data)
        rng = xmax - xmin
        if rng < 1e-3:
            rng = 1e-3
        xmin -= 0.01*rng
        xmax += 0.01*rng
        ax.plot([xmin, xmax], [xmin, xmax], color='#1c1c14')
        
        ax.plot(dft_data, ce_data, 'o', mfc='none', color='#742d18', markersize=2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel(info['xlabel'])
        ax.set_ylabel(info['ylabel'])
        fig.tight_layout()
        fig.savefig(info['fname'])
        fig.savefig(info['fname'].rpartition('.')[0] + '.png')
        print(f"Figure saved to {info['fname']}")



if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        fit()
    elif sys.argv[1] == 'plot':
        plot()
    elif sys.argv[1] == "fitvol":
        fit(fit_type="volume")
    elif sys.argv[1] == "plotvol":
        plotvol()
