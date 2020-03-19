import matplotlib as mpl
mpl.rcParams.update({'font.size': 18, 'axes.unicode_minus': False, 'svg.fonttype': 'none'})
from matplotlib import pyplot as plt
import json
import numpy as np


def plot_coeff_dist(highscore, num):
    data = np.loadtxt("agpt_data.csv", delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    print(X.shape)

    with open("agpt_data.csv", 'r') as infile:
        header = infile.readline()
        header = header.replace('#', '')
        header = header.replace(' ', '')
        sizes = [int(x[1]) for x in header.split(',')[:-1]]

    score_sel = [(x['Score'], x['Selection']) for x in highscore]
    score_sel.sort()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    all_coeff = [[] for _ in range(X.shape[1])]
    for item in score_sel[:num]:
        sel = item[1]
        X_sel = X[:, sel]
        u, s, vh = np.linalg.svd(X_sel, full_matrices=False)
        V = vh.T
        diag_item = np.zeros_like(s)
        mask = np.abs(s) > 1e-10
        diag_item[mask] = 1.0/s[mask]
        coeff = V.dot(np.diag(diag_item)).dot(u.T).dot(y)
        ax.plot(sel, coeff, 'o', color='grey', alpha=0.1)
        pred = X_sel.dot(coeff)
        rmse = np.sqrt(np.mean((pred - y)**2))
        print(rmse)

        for i in range(len(sel)):
            all_coeff[sel[i]].append(coeff[i])
    ax.axhline(0.0, ls='--')

    fig_mean = plt.figure()
    ax_mean = fig_mean.add_subplot(1, 1, 1)
    ax_mean.axhline(0.0, ls='--')

    colors = {
        2: '#766868',
        3: '#b63119',
        4: '#cb9f52'
    }
    for i, v in enumerate(all_coeff):
        if sizes[i] <= 1:
            continue
        mean = np.mean(v)*1000.0
        std = np.std(v)*1000.0
        ax_mean.errorbar(i, [mean], yerr=std, color=colors[sizes[i]], capsize=2,
                         marker='o', markersize=1, mfc='none')
    ax_mean.spines['right'].set_visible(False)
    ax_mean.spines['top'].set_visible(False)
    ax_mean.set_xlabel("Feature no.")
    ax_mean.set_ylabel("ECI (meV/atom)")
    return fig


def main():
    fname = "saAgPtSearch.json"

    scores = []
    with open(fname, 'r') as infile:
        data = json.load(infile)

    for item in data['Items']:
        scores.append(item["Score"])

    plot_coeff_dist(data['Items'], 40)
    scores.sort()
    scores = np.array(scores)
    print(scores)
    scores -= np.min(scores)

    w = np.exp(-scores)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(w, marker='o', mfc='none', color='#766868')
    ax.set_ylim([1e-2, 1.1])
    ax.set_xlabel("Model no.")
    ax.set_ylabel("\$\exp(AICC_\{min\} - AICC)\$")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


if __name__ == '__main__':
    main()
