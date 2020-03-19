from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge, LassoCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from matplotlib import pyplot as plt
from random import shuffle

#FNAME = 'data/almgsiX_1621.csv'
FNAME = 'data/almgsi_binary_linear.csv'

BF = [{'Al': 1.2649110640673518, 'Mg': -1.2649110640673518, 'Si': 0.6324555320336759, 'X': -0.6324555320336759},
      {'Al': 0.9999999999999998, 'Mg': 0.9999999999999998, 'Si': -1.0, 'X': -1.0},
      {'Al': 0.632455532033676, 'Mg': -0.632455532033676, 'Si': -1.2649110640673518, 'X': 1.2649110640673518}]

def get_training_set():
    data = np.loadtxt(FNAME, delimiter=',')
    energy = data[:, -1]
    energy -= np.mean(energy)
    X = data[:, :-1]
    return X, energy

def get_names():
    with open(FNAME, 'r') as infile:
        header = infile.readline()
    
    header = header.replace('#', '')
    header = header.replace(' ', '')
    return header.split(',')


def fit_rfe():
    X_train, y_train = get_training_set()
    scaler = StandardScaler()
    scaler.fit(X_train)
    X = scaler.transform(X_train)

    estimator = RFECV(Ridge(1e-7, fit_intercept=False), cv=5, verbose=1, step=1)
    estimator.fit(X, y_train)
    #print(estimator.grid_scores_)
    pred = estimator.estimator_.predict(X[:, estimator.support_])
    mse = np.sqrt(np.mean((pred-y_train)**2))
    R_sq = np.max(estimator.grid_scores_)
    cv = np.sqrt((1.0 - R_sq)*np.sum(y_train**2)/len(y_train))
    plt.plot(estimator.grid_scores_)
    plt.show()
    print(mse, cv)

def fit_lasso():
    X_train, y_train = get_training_set()
    X_train = X_train[:, :2000]
    scaler = StandardScaler()
    scaler.fit(X_train)
    X = scaler.transform(X_train)

    estimator = LassoCV(eps=1e-5, fit_intercept=False, cv=5)

    estimator.fit(X, y_train)
    cvs = np.sqrt(np.mean(estimator.mse_path_, axis=1))
    print(np.min(cvs))

def mse(y1, y2):
    return np.sqrt(np.mean((y1-y2)**2))

def fit_lasso_rf():
    X_train, y_train = get_training_set()
    scaler = StandardScaler()
    scaler.fit(X_train)
    X = scaler.transform(X_train)

    indices = list(range(X_train.shape[0]))
    shuffle(indices)
    split_N = int(len(indices)/2)
    X_lasso = X[indices[:split_N], :]
    y_lasso = y_train[indices[:split_N]]
    estimator = LassoCV(eps=1e-5, fit_intercept=False, cv=5)
    estimator.fit(X_lasso, y_lasso)
    cvs = np.sqrt(np.mean(estimator.mse_path_, axis=1))
    selected = np.nonzero(estimator.coef_)[0]
    print("Num. selected: {}".format(len(selected)))
    print(np.min(cvs))

    X_rf = X[indices[split_N:], :]
    y_rf = y_train[indices[split_N:]]
    pred = estimator.predict(X_rf)
    print(mse(pred, y_rf))
    
    residuals = y_rf - pred
    forest = RandomForestRegressor(verbose=1)
    X_rf = X_rf[:, selected]
    forest.fit(X_rf, residuals)

    pred_in_sample = forest.predict(X_rf)
    pred_out_of_sample = forest.predict(X_lasso[:, selected])
    mse_in = mse(pred_in_sample, residuals)
    mse_out = mse(pred_out_of_sample, y_lasso - estimator.predict(X_lasso))
    print(mse_in, mse_out)

    # Total MSE
    pred_lasso = estimator.predict(X)
    pred = pred_lasso + forest.predict(X[:, selected])
    print(mse(pred_lasso, y_train))
    print(mse(pred, y_train))

def lasso_filter():
    from clease.tools import singlets2conc
    X_train, y_train = get_training_set()
    concs = singlets2conc(BF, X_train[:, 1:4])

    ranges = [{
        'element': 'Al',
        'range': [0, 0.5]
    },{
        'element': 'Al',
        'range': [0.25, 0.75]
    },{
        'element': 'Al',
        'range': [0.5, 1.0]
    }]

    for group in ranges:
        cmin = group['range'][0]
        cmax = group['range'][1]
        selected = [i for i, v in enumerate(concs) if v[group['element']] > cmin and v[group['element']] < cmax]
        X = X_train[selected, :]
        y = y_train[selected]

        reg = LassoCV(eps=1e-5)
        reg.fit(X, y)
        cvs = np.sqrt(np.mean(reg.mse_path_, axis=1))
        print(np.min(cvs))

def physical_ridge():
    from clease import PhysicalRidge
    from clease.physical_ridge import random_cv_hyper_opt
    X_train, y_train = get_training_set()
    names = get_names()[:-1]

    #X_train = X_train[:, :1000]
    #names = names[:1000]
    regressor = PhysicalRidge(lamb_size=1e-6, lamb_dia=1e-6,
                              dia_decay='exponential',
                              size_decay='exponential')

    regressor.sizes_from_names(names)
    regressor.diameters_from_names(names)

    params = {
        'lamb_size': np.logspace(-9, 3, 100).tolist(),
        'lamb_dia': np.logspace(-9, 3, 100).tolist(),
        'dia_decay': ['linear', 'exponential'],
        'size_decay': ['linear', 'exponential']
    }

    res = random_cv_hyper_opt(regressor, params, X_train, y_train, cv=5, num_trials=10000)
    print(res['params'][:10])
    print(res['cvs'][:10])
    best_param = res['best_params']
    regressor.lamb_dia = best_param['lamb_dia']
    regressor.lamb_size = best_param['lamb_size']
    regressor.size_decay = best_param['size_decay']
    regressor.dia_decay = best_param['dia_decay']
    coeff = regressor.fit(X_train, y_train)
    plt.plot(coeff, marker='x')
    plt.show()

physical_ridge()
#fit_rfe()
#fit_lasso()
#fit_lasso_rf()
#lasso_filter()

