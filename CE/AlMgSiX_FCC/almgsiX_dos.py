from ase.db import connect
from matplotlib import pyplot as plt
import numpy as np
import json
from clease import settingFromJSON
from clease.calculator import attach_calculator
from random import choice
import time
from itertools import product
from clease.tools import singlets2conc
from sklearn.linear_model import LassoCV, RidgeCV, lasso_path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from random import shuffle

def explore():
    eci_file = 'data/almgsiX_eci.json'
    with open(eci_file, 'r') as infile:
        eci = json.load(infile)

    settings = settingFromJSON('almgsixSettings.json')
    
    atoms = settings.atoms*(5, 5, 5)
    atoms = attach_calculator(setting=settings, atoms=atoms, eci=eci)

    num_structs = 10*len(atoms)
    energies = []
    symbs = ['Al', 'Mg', 'Si', 'X']
    now = time.time()
    for i in range(num_structs):
        if time.time() - now > 30:
            print("{} of {}".format(i, num_structs))
            now = time.time()
        idx = np.random.randint(0, high=len(atoms))
        symb = choice(symbs)
        atoms[idx].symbol = symb
        energies.append(atoms.get_potential_energy()/len(atoms))
    np.savetxt("data/energies.txt", energies)

def main():
    db_name = 'almgsiX_dft2.db'
    db = connect(db_name)
    energies = []
    chem_pot = {'Al': 0.0, 'Mg': 0.0, 'Si': 0.0}
    for row in db.select():
        energy = row.get('energy', None)
        count = row.count_atoms()
        if energy is not None:
            energy -= sum(count[k]*chem_pot[k]/row.natoms for k in count.keys())
            energies.append(energy/row.natoms)
    
    hist, edges = np.histogram(energies, bins=50)
    print(np.sum(hist))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(edges[1:], hist, drawstyle='steps')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    x = np.sort(energies)
    y = np.linspace(0.0, 1.0, len(x))
    ax2.plot(x, y, drawstyle='steps', label='DFT')

    ce = np.loadtxt("data/energies.txt")
    x = np.sort(ce)
    y = np.linspace(0.0, 1.0, len(x))
    ax2.plot(x, y, drawstyle='steps', label='CE')
    plt.show()

def interpolator(x):
    return 3.0*x**2 - 2.0*x**3

def restricted_CE():
    data = np.loadtxt('data/almgsiX_data.csv', delimiter=',')
    energy = data[:, -1]
    energy -= np.mean(energy)
    X = data[:, :-1]
    # singlets = data[:, 1:4]
    # settings = settingFromJSON('almgsixSettings.json')

    coeff = np.zeros(X.shape[1])
    cv_score = 0.0
    for _ in range(2):
        model = LassoCV(eps=1e-5, tol=1e-4, fit_intercept=False)
        model.fit(X, energy)
        cvs = np.mean(np.sqrt(model.mse_path_), axis=1)
        coeff += model.coef_
        energy -= model.predict(X)
        cv_score = 1000.0*np.min(cvs)

    print("Min CV: {} meV/atom".format(cv_score))
    alphas, coefs, dual_gaps, n_iter = lasso_path(X, energy, method='lasso')
    print(coefs)

    np.savetxt("data/coeff_all.txt", coeff)

def random_forest_residuals():
    data = np.loadtxt('data/almgsiX_data.csv', delimiter=',')
    energy = data[:, -1]
    energy -= np.min(energy)
    X = data[:, :-1]
    coeff = np.loadtxt("data/coeff_all.txt")
    res = energy - X.dot(coeff)

    reg = RandomForestRegressor(n_estimators=100)
    reg.fit(X, res)
    pred = reg.predict(X)
    mse = np.sqrt(np.mean((pred - res)**2))*1000.0
    print(mse)




def show_fit():
    data = np.loadtxt('data/almgsiX_data.csv', delimiter=',')
    energy = data[:, -1]
    energy -= np.mean(energy)
    X = data[:, :-1] 
    singlets = data[:, 1:4]
    settings = settingFromJSON('almgsixSettings.json')
    concs = singlets2conc(settings.basis_functions, singlets)

    coeff_all = np.loadtxt("data/coeff_all.txt")
    coeff_alrich = np.loadtxt("data/coeff_alrich.txt")
    coeff_al_low = np.loadtxt("data/coeff_al_low.txt")

    c_al = np.array([v['Al'] for v in concs])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    pred_all = X.dot(coeff_all)
    pred_alrich = X.dot(coeff_alrich)
    pred_al_low = X.dot(coeff_al_low)
    weights = interpolator(c_al)

    pred_sum = pred_alrich*weights + pred_al_low*(1.0 - weights)
    print(energy - pred_all)

    ax.plot(c_al, pred_all - energy, 'o', mfc='none')
    ax.plot(c_al, pred_alrich - energy, 'x')
    ax.plot(c_al, pred_al_low - energy, 'v', mfc='none')
    ax.plot(c_al, pred_sum - energy, '^', mfc='none')
    plt.show()

def fit_eci():
    data = np.loadtxt('data/almgsiX_data.csv', delimiter=',')
    energy = data[:, -1]
    X = data[:, :-1]
    #energy -= np.mean(energy)
    indices = list(range(X.shape[0]))
    shuffle(indices)
    N = int(0.8*len(indices))
    train = indices[:N]
    test = indices[N:]
    X_test = X[test, :]
    energy_test = energy[test]

    # Remove test data
    X = X[train, :]
    energy = energy[train]
   

    singlets = data[:, 1:4]
    settings = settingFromJSON('almgsixSettings.json')
    concs = singlets2conc(settings.basis_functions, singlets)
    conc_array = np.array([[v for v in x.values()] for x in concs])
    
    indicators = conc_array[:, :-1]
    coeffs = []
    weights = []
    for types in product(range(0, 2), repeat=3):
        multiplicator = np.ones(data.shape[0])
        for i, t in enumerate(types):
            if t == 0:
                multiplicator *= interpolator(indicators[:, i])
            else:
                multiplicator *= (1.0-interpolator(indicators[:, i]))
        mdata = np.zeros_like(X)
        for i in range(X.shape[1]):
            mdata[:, i] = X[:, i]*multiplicator[train]
        
        model = LassoCV(eps=1e-6, cv=5)
        #model = RidgeCV(alphas=np.logspace(-12, -5, 50))
        energy_fit = energy*multiplicator[train]
        model.fit(mdata, energy_fit)
        coeffs.append(model.coef_)
        weights.append(multiplicator)

    pred = np.zeros_like(energy)
    pred_test = np.zeros_like(energy_test)
    print(X_test.shape)
    for i in range(len(coeffs)):
        pred += X.dot(coeffs[i])*weights[i][train]
        pred_test += X_test.dot(coeffs[i])*weights[i][test]
    
    rmse = np.sqrt(np.mean((energy - pred)**2))
    rmse_test = np.sqrt(np.mean((energy_test - pred_test)**2))
    print("MSE: {} meV/atom".format(rmse*1000.0))
    print("MSE test: {} meV/atom".format(rmse_test*1000.0))

    



def structure_impact():
    eci_file = 'data/almgsiX_eci.json'
    with open(eci_file, 'r') as infile:
        eci = json.load(infile)

    print(eci)
    settings = settingFromJSON('almgsixSettings.json')
    print(settings.basis_functions)
    dE = 0.011 # 11 meV/atom
    ecis = np.array([[v for k, v in eci.items() if k.startswith(('c1', 'c2', 'c3', 'c4'))]])
    x, _, _, _ = np.linalg.lstsq(ecis, [dE])
    print(np.sum(np.abs(x)))
    print(dE/np.max(np.abs(ecis)))
    print(dE/np.median(np.abs(ecis)))


#show_fit()
#restricted_CE()
random_forest_residuals()
#fit_eci()
#structure_impact()
#main()
#explore()
