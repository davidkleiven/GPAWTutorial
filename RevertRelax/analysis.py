import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'svg.fonttype': 'none', 'font.family': 'serif'})
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from itertools import combinations


fname = "data/map_results.csv"

FEAT_MAP = {
    'max_dist': 'Max. dist',
    'min_dist': 'Min. dist',
    'mean_dist': 'Mean dist',
    'std_dist': 'Std. dist',
    'form_corr': 'F-norm $\\rho$',
    'skew_dist': 'Skewness',
    'kurtosis_dist': 'Kurtosis',
    'hyd_strain': 'Hyd. strain',
    'fnorm_dev_strain': 'Dev. strain',
    'vac_conc': 'Vac. conc'
}

def sigmoid(x, alpha, x0):
    return 1.0/(1.0 + np.exp(alpha*(x-x0)))

def explore():
    data = pd.read_csv(fname)
    print(data.columns)
    fig, axs = plt.subplots(3, 4, constrained_layout=True)
    plt.tight_layout()
    for i, col in enumerate(data.columns[2:]):
        ax = axs.flat[i]

        ax.plot(data[col], data['success'], 'o', mfc='none', color='#1c1c14')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        x0 = np.mean(data[col])
        popt, _ = curve_fit(sigmoid, data[col], data['success'], p0=[0.0, x0])
        ax.set_xlabel(col)
        x = np.linspace(np.min(data[col]), np.max(data[col]), 100)
        ax.plot(x, sigmoid(x, popt[0], popt[1]), color='#742d18')
    
    num_sucess = np.sum(data['success'])
    tot_num = len(data['success'])
    print(f"Sucess rate: {num_sucess/tot_num}")


def cross_corr():
    data = pd.read_csv(fname)
    
    plt.tight_layout()
    
    features = data.columns
    exclude = ['success', 'Formula', 'system']
    features = [x for x in features if x not in exclude]
    N = len(features)
    totNum = int(N*(N-1)/2)
    ncol = 5
    nrow = int(totNum/ncol)
    if nrow*ncol < totNum:
        nrow += 1
    fig, axs = plt.subplots(nrow, ncol, constrained_layout=True)
    counter = 0
    for comb in combinations(features, r=2):
        sucess_x = []
        sucess_y = []
        fail_x = []
        fail_y = []
        for _, row in data.iterrows():
            if row['success']:
                sucess_x.append(row[comb[0]])
                sucess_y.append(row[comb[1]])
            else:
                fail_x.append(row[comb[0]])
                fail_y.append(row[comb[1]])
        ax = axs.flat[counter]
        ax.set_xlabel(FEAT_MAP.get(comb[0], comb[0]))
        ax.set_ylabel(FEAT_MAP.get(comb[1], comb[1]))
        ax.plot(sucess_x, sucess_y, 'o', mfc='none', color='#1c1c14', markersize=2)
        ax.plot(fail_x, fail_y, 'v', mfc='none', color='#742d18', markersize=2)
        counter += 1
    fig.set_size_inches(12, 20)
    fig.savefig("cross_corr.png")


def displacement_distribution():
    data_suc = np.loadtxt("data/displacement_dist_success.csv")
    data_fail = np.loadtxt("data/displacement_dist_failure.csv")
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    hist, edges = np.histogram(data_suc, bins=40)
    ax.plot(edges[1:], hist/np.sum(hist), drawstyle='steps')

    hist, edges = np.histogram(data_fail, bins=40)
    ax.plot(edges[1:], hist/np.sum(hist), drawstyle='steps')
    ax.set_xlabel("Displacement")
    ax.set_ylabel("Probability Density Function")

    ax2 = fig.add_subplot(1, 2, 2)
    P = np.linspace(0.0, 1.0, len(data_suc))
    ax2.plot(np.sort(data_suc), P, drawstyle='steps')
    
    P = np.linspace(0.0, 1.0, len(data_fail))
    ax2.plot(np.sort(data_fail), P, drawstyle='steps')
    ax2.set_xlabel("Displacement")
    ax2.set_ylabel("Cummulative distribution")


def fit_logit():
    data = pd.read_csv(fname)
    print(data.shape)

    model = LogisticRegression() 
    X = np.zeros((data.shape[0], 2))
    X[:, 0] = data['max_dist']
    X[:, 1] = data['form_corr']
    model.fit(X, data['success'])
    pred = model.predict(X)
    num_correct = np.count_nonzero(pred == data['success'])
    print(f"Correctly predicted: {num_correct/len(pred)}")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    prob_success = model.predict_proba(X)[:, 1]
    threshold = np.linspace(0.1, 0.99, 100)
    false_pos = []
    false_neg = []
    recovered = []
    for i in range(len(threshold)):
        mask = prob_success > threshold[i]
        num_rec = np.sum(mask)
        recovered.append(num_rec/len(prob_success))
        num_false_positive = num_rec - np.sum(data['success'][mask])
        num_false_negative = np.sum(data['success'][~mask])
        false_pos.append(num_false_positive/num_rec)
        false_neg.append(num_false_negative/(len(prob_success) - num_rec))

    ax.plot(threshold, false_pos, 'o', mfc='none', color='#742d18')
    ax.plot(threshold, false_neg, 'v', mfc='none', color='#453524')
    ax2 = ax.twinx()
    ax2.plot(threshold, recovered, color='#1c1c14')
    ax.set_yscale('log')
    ax.set_xlabel("Threshold")
    ax.set_ylabel("False positive/negative")
    ax2.set_ylabel("Fraction of structures recovered")
    
def plot_decision_pairs():
    data = pd.read_csv(fname)
    columns = ['max_dist', 'form_corr', 'std_dist', 'mean_dist',
               'skew_dist', 'kurtosis_dist', 'min_dist']
    X = data.as_matrix(columns=columns)
    y = data['success']
    N = X.shape[1]*(X.shape[1] + 1)/2
    nrows = int(N/3)
    counter = 1
    fig, axs = plt.subplots(nrows, 3, constrained_layout=True)
    #fig.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    for pair in combinations(range(X.shape[1]), r=2):
        print(pair)
        X_fit = X[:, pair]

        clf = DecisionTreeClassifier().fit(X_fit, y)
        #ax = fig.add_subplot(nrows, 3, counter)
        ax = axs.flat[counter]
        x_min, x_max = X_fit[:, 0].min() - 1, X_fit[:, 0].max() + 1
        y_min, y_max = X_fit[:, 1].min() - 1, X_fit[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50, endpoint=True),
                         np.linspace(y_min, y_max, 50, endpoint=True))
        #ax.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = ax.contourf(xx, yy, Z, cmap='coolwarm')

        # Add data
        idx = np.where(y == 1)[0]
        print(idx)
        ax.plot(X_fit[idx, 0], X_fit[idx, 1], 'o', mfc='none', markersize=2, alpha=0.5)
        idx = np.where(y == 0)[0]
        ax.plot(X_fit[idx, 0], X_fit[idx, 1], 'v', mfc='none', markersize=2, alpha=0.5)
        ax.set_xlabel(columns[pair[0]])
        ax.set_ylabel(columns[pair[1]])
        counter += 1

def plot_feature_importance(names, importance):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    names = [FEAT_MAP.get(n, n) for n in names]
    ax.barh(names, importance, color='#1c1d14')
    ax.set_xlabel("Importance")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def false_positive(rf, X_train, y_train, X_test, y_test, type='falsepos'):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)

    fig2 = plt.figure(figsize=(4, 3))
    ax2 = fig2.add_subplot(1, 1, 1)

    dsets = [{
        'X': X_train,
        'y': y_train,
        'ls': '-',
        'name': 'Train'
    },{
        'X': X_test,
        'y': y_test,
        'ls': '--',
        'name': 'Validation'
    }]

    for dset in dsets:
        y = dset['y']
        X = dset['X']
        pred = np.zeros(len(y))
        all_pred = rf.predict(X)
        for tree in rf.estimators_:
            pred += tree.predict(X)
        pred /= len(rf.estimators_)
        mask = all_pred == 1
        y_pred_sucess = y[mask]
        x = pred[mask][y_pred_sucess==0]
        hist, edges = np.histogram(x, bins=10)
        w = 0.8*(edges[1] - edges[0])
        ax.bar(edges[1:]-w/2, hist, width=w, color='#742d18', label="False positive")

        y_pred_not_sucess = y[~mask]
        x = pred[~mask][y_pred_not_sucess==1]
        # hist, edges = np.histogram(x, bins=10)
        # w = 0.8*(edges[1] - edges[0])
        # ax.bar(edges[1:]-w/2, hist, width=w, color='#1c1c14', label="False negative")
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # ax.set_xlabel("Relative number of votes")
        # ax.set_ylabel("Num. occurences")
        # ax.legend(frameon=False)

        # Thresholding
        thresholds = np.linspace(0.01, 0.99, 100)
        frac_rec = []
        frac_fp = []
        frac_fn = []
        for t in thresholds:
            mask = pred > t
            num_rec = np.sum(mask)
            num_not_rec = len(pred) - num_rec
            num_false_pos = np.count_nonzero(y[mask] == 0)
            num_false_neg = np.count_nonzero(y[~mask])

            frac_rec.append(num_rec/len(pred))
            frac_fp.append(num_false_pos/num_rec)
            frac_fn.append(num_false_neg/num_not_rec)

        ax2.plot(thresholds, frac_fp, ls=dset['ls'], color='#1c1c14')
        ax2.plot(thresholds, frac_fn, ls=dset['ls'], color='#742d18')
        #ax2.set_yscale('log')
        ax2.plot(thresholds, frac_rec, ls=dset['ls'], color='#453524', label=dset['name'])
    ax2.set_xlabel("Relative number of votes")
    ax2.set_ylabel("Relative occurence")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.legend(frameon=False)



def fit_ada_boost():
    data = pd.read_csv(fname)
    columns = ['max_dist', 'form_corr', 'std_dist', 'mean_dist',
               'skew_dist', 'kurtosis_dist', 'min_dist', 'vac_conc']
    X = data.as_matrix(columns=columns)
    y = data['success']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    params = {
        'n_estimators': [70],
        'max_depth': [4]
    }
    #classifier = AdaBoostClassifier()
    classifier = RandomForestClassifier(n_estimators=3)
    cv_selection = GridSearchCV(classifier, params, cv=20)
    cv_selection.fit(X_train, y_train)

    importance = cv_selection.best_estimator_.feature_importances_
    srt_idx = np.argsort(importance).tolist()
    sorted_columns = [columns[i] for i in srt_idx]
    sorted_importance = [importance[i] for i in srt_idx]
    plot_feature_importance(sorted_columns, sorted_importance)
    #classifier.fit(X, y)
    scores = cross_val_score(classifier, X_train, y_train, cv=5)
    print(scores.mean())
    print(cv_selection.best_params_)


    pred = cv_selection.predict(X_train)
    #print(prob[:, 1])
    
    num_recovered = np.sum(pred)
    num_not_recovered = len(pred) - num_recovered
    mask = pred == 1

    # Count the number 0 where pred equals 1
    num_false_pos = np.count_nonzero(y_train[mask] == 0)

    # Count number of ones where pred equals 0
    num_false_neg = np.count_nonzero(y_train[~mask])
    print(f"Fraction recovered: {num_recovered/len(pred)}")
    print(f"Fraction false positive: {num_false_pos/num_recovered}")
    print(f"Fraction false negative: {num_false_neg/num_not_recovered}")
    false_positive(cv_selection.best_estimator_, X, y, X_test, y_test)

def success_rates():
    data = pd.read_csv(fname)
    num_suc = {}
    tot_num = {}
    for i, item in enumerate(data['success']):
        system = data['system'][i]
        if item == 1:
            N = num_suc.get(system, 0)
            N += 1
            num_suc[system] = N
        N = tot_num.get(system, 0)
        N += 1
        tot_num[system] = N

    for k in tot_num.keys():
        print(f"{k}: {num_suc[k]/tot_num[k]} {tot_num[k]}")

    tot_num_suc = sum(num_suc.values())
    tot_num_data = sum(tot_num.values())
    print(tot_num_suc/tot_num_data, tot_num_data)
#explore()
#cross_corr()
fit_ada_boost()
#success_rates()
#plot_decision_pairs()
#displacement_distribution()
#fit_logit()
plt.show()