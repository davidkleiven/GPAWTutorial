import matplotlib as mpl
mpl.rcParams.update({'font.size': 18, 'svg.fonttype': 'none', 'axes.unicode_minus': False})
from matplotlib import pyplot as plt
import json
import numpy as np

FNAME = 'nestedlassoTrain.json'


def extract_unique_models(fname):
    """
    Extracts the unique models from nested lasso
    """
    with open(fname, 'r') as infile:
        data = json.load(infile)

    selections = []
    coeffs = []
    aicc = []
    bic = []
    for item in data['Paths']:
        for i, node in enumerate(item['Nodes']):
            if node['Selection'] not in selections:
                selections.append(node['Selection'])
                coeffs.append(node['Coeff'])
                aicc.append(item['Aicc'][i])
                bic.append(item['Bic'][i])
    return selections, coeffs, aicc, bic


def plot_aicc_bic_per_iteration(fname):
    with open(fname, 'r') as infile:
        data = json.load(infile)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for item in data['Paths']:
        lambs = [n['Lamb'] for n in item['Nodes']]
        lambs = np.log10(lambs)
        ax.plot(lambs, item['Aicc'], color='grey', alpha=0.4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('\$\log \lambda\$')
    ax.set_ylabel("AICc")
    plt.show()


def pred_validation(fname):
    s, c, a, b = extract_unique_models(fname)

    # Extract first model
    with open(fname, 'r') as infile:
        paths = json.load(infile)
    best1 = np.argmin(paths['Paths'][0]['Aicc'])
    s1 = paths['Paths'][0]['Nodes'][best1]['Selection']
    c1 = paths['Paths'][0]['Nodes'][best1]['Coeff']

    data = np.loadtxt('agpt_validate.csv', delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]

    best_aic = np.argmin(a)
    pred = X[:, s[best_aic]].dot(c[best_aic])
    mse = np.sqrt(np.mean((y - pred)**2))*1000
    print("MSE best AIC: {} meV/atom".format(mse))

    best_bic = np.argmin(b)
    pred = X[:, s[best_bic]].dot(c[best_bic])
    mse = np.sqrt(np.mean((y - pred)**2))*1000
    print("MSE best BIC: {} meV/atom".format(mse))

    a -= np.min(a)
    w = np.exp(-a)
    w[w < 0.01] = 0.0
    w /= np.sum(w)

    pred_aic = np.zeros(len(y))
    for i in range(len(s)):
        pred_aic += w[i]*X[:, s[i]].dot(c[i])

    mse = np.sqrt(np.mean((y - pred_aic)**2))*1000
    print("MSE avg AIC: {} meV/atom".format(mse))

    b -= np.min(b)
    w = np.exp(-b)
    w[w < 0.01] = 0.0
    w /= np.sum(w)

    pred_bic = np.zeros(len(y))
    for i in range(len(s)):
        pred_bic += w[i]*X[:, s[i]].dot(c[i])

    mse = np.sqrt(np.mean((y - pred_bic)**2))*1000
    print("MSE avg BIC: {} meV/atom".format(mse))

    pred = X[:, s1].dot(c1)
    mse = np.sqrt(np.mean((y - pred)**2))*1000
    print('MSE min AIC 1 gen {} meV/atom'.format(mse))

#plot_aicc_bic_per_iteration(FNAME)
pred_validation(FNAME)
