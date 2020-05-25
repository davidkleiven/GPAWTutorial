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
from random import shuffle, choice
from ase.calculators.emt import EMT
from clease.tools import wrap_and_sort_by_position
from copy import deepcopy

db_local = "data/structures_with_bulk.db"
settings_file = "data/settings_almgsiX_voldev.json"
#settings_file = "data/aucu_emt_demo_settings.json"
table = "binary_linear_cf"

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

def plot_dBdP():
    from clease.calculator import CleaseVolDep
    from clease.data_manager import CorrelationFunctionGetter
    with open("data/voldep_fit_resultenergy.json", 'r') as infile:
        data = json.load(infile)
    eci_eng = data['eci']
    with open("data/voldep_fit_resultvolume.json", 'r') as infile:
        data_vol = json.load(infile)
    eci_vol = data_vol['eci']
    settings = settingsFromJSON(settings_file)
    calc = CleaseVolDep(settings, eci_eng, eci_vol)

    corr_func = CorrelationFunctionGetter(
        settings.db_name, table, settings.all_cf_names
    )
    ids = set()
    db = connect(settings.db_name)
    dBdP_db = []
    for row in db.select([('struct_type', '=', 'initial')]):
        ids.add(row.id)
        dBdP_db.append(row.dBdP)
    print(ids)
    cf = corr_func(ids)
    atoms = settings.atoms.copy()
    atoms.set_calculator(calc)

    dBdP = []
    for r in range(0, cf.shape[0]):
        cf_dict = {k: v for k, v in zip(corr_func.names, list(cf[r, :]))}
        dBdP.append(calc.get_dBdP(cf_dict))
    
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dBdP_db, dBdP, 'o', mfc='none', markersize=2, color='#742d18')
    xmin = min(dBdP)
    xmax = max(dBdP)
    rng = xmax - xmin
    xmin -= 0.01*rng
    xmax += 0.01*rng
    ax.plot([xmin, xmax], [xmin, xmax], color='#1c1c14')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel("$(\u2202 B/\u2202 P)_0$ EMT")
    ax.set_ylabel("$(\u2202 B/\u2202 P)_0$ CE")
    fig.tight_layout()
    fig.savefig("fig/dBdP_proper.png")


def fit(fit_type="energy"):
    settings = settingsFromJSON(settings_file)
    #print(settings.multiplicity_factor)
    #exit()
    cf_names = settings.all_cf_names
    #cf_names = None
    cf_names = [n for n in cf_names if int(n[1]) < 3]
    print(cf_names)
    order = 5
    if fit_type == "energy":
        data = CorrelationFunctionGetterVolDepECI(
            settings.db_name,
            table,
            cf_names,
            order=order,
            properties=['energy', 'bulk_mod', 'dBdP'],
            cf_order=2
        )
        pressure_data = deepcopy(data)
        pressure_data.properties=('pressure')
    elif fit_type == "volume":
        data = CorrFuncVolumeDataManager(
             settings.db_name,
            table,
            cf_names
        )

    skip_groups = [5, 13, 42, 71, 84, 1490300799, 3797755424]
    scond = [('struct_type', '=', 'initial')]
    for sg in skip_groups:
        scond.append(('name', '!=', f'group{sg}'))
    X, y = data.get_data(scond)

    Xp, yp = pressure_data.get_data(scond)
    Np = int(Xp.shape[0]/2)
    Xp = Xp[Np:, :]
    yp = yp[Np:]

    N = X.shape[0]

    #X[int(N/3):int(2*N/3)] *= 1000.0
    #y[int(N/3):] *= 1000.0

    regressor = PhysicalRidge(normalize=False)
    regressor.add_constraint(Xp, yp)
    groups = data.groups()
    #regressor.sizes = [int(n[1]) for n in data._feat_names]
    regressor.sizes = [sum(int(n[1]) for n in m.split('*')) for m in data._feat_names]
    #print(regressor.sizes)
    #exit()
    prog = re.compile(r"d(\d+)")
    regressor.diameters = []
    for cf_name in data._feat_names:
        result = prog.findall(cf_name)
        if not result:
            regressor.diameters.append(0.0)
        else:
            regressor.diameters.append(sum(float(x) for x in result))

    params = {
            'lamb_dia': np.logspace(-6, 6, 5000).tolist(),
            'lamb_size': np.logspace(-6, 6, 5000).tolist(),
            'size_decay': ['linear', 'exponential', 'poly2', 'poly4', 'poly6'],
            'dia_decay': ['linear', 'exponential', 'poly2', 'poly4', 'poly6']
        }
    res = random_cv_hyper_opt(regressor, params, X, y, cv=5, num_trials=100, groups=groups)

    outfile = f"data/voldep_fit_result{fit_type}.json"
    data = {
        'names': cf_names,
        'coeff': res['best_coeffs'].tolist(),
        'X': X.tolist(),
        'y': y.tolist(),
        'cv': res['best_cv'],
        'eci': {n: c for n, c in zip(data._feat_names, res['best_coeffs'])},
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

    fig = plt.figure(figsize=(4, 3))
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
    ax.set_xlabel("EMT volume (Å$^3$)")
    ax.set_ylabel("CE volume (Å$^3$)")
    fig.tight_layout()
    fig.savefig("fig/volume_fit.pdf")
    fig.savefig("fig/volume_fit.png")

    
def plot():
    from clease.calculator import CleaseVolDep
    with open(f"data/voldep_fit_resultenergy.json", 'r') as infile:
        data = json.load(infile)
    X = np.array(data['X'])
    y = np.array(data['y'])
    coeff = data['coeff']

    settings = settingsFromJSON(settings_file)
    eci_eng = data['eci']
    with open("data/voldep_fit_resultvolume.json", 'r') as infile:
        data_vol = json.load(infile)
    eci_vol = data_vol['eci']
    calc = CleaseVolDep(settings, eci_eng, eci_vol)

    atoms = settings.atoms.copy()*(2, 2, 2)
    atoms = wrap_and_sort_by_position(atoms)
    #atoms.set_calculator(calc)

    plots_info = [
        {
            'ylabel': 'Energy CE (eV/atom)',
            'xlabel': 'Energy EMT (eV/atom)',
            'fname': 'fig/energy_pred.pdf',
            'factor': 1.0,
        },
        {
            'ylabel': 'Bulk mod. CE (GPa)',
            'xlabel': 'Bulk mod. EMT (GPa)',
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
        start = int(i*len(y)/3)
        end = int((i+1)*len(y)/3)
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

        # Add out of sample
        # if 'Energy' in info['xlabel']:
        #     out_pred = []
        #     out_ex = []
        #     for i in range(40):
        #         for atom in atoms:
        #             atom.symbol = choice(settings.concentration.basis_elements[0])
        #         emt_calc = EMT()
        #         atoms_cpy = atoms.copy()
        #         atoms_cpy.set_calculator(emt_calc)
        #         out_ex.append(atoms.get_potential_energy()/len(atoms))
        #         out_pred.append(atoms.get_potential_energy()/len(atoms))

            #ax.plot(out_ex, out_pred, 'v')
            
        ax.set_xlabel(info['xlabel'])
        ax.set_ylabel(info['ylabel'])
        if 'Pressure' in info['xlabel']:
            ax.set_xticks([0.0])
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
    elif sys.argv[1] == 'plotdbdp':
        plot_dBdP()
