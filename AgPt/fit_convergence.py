import matplotlib as mpl
mpl.rcParams.update({'font.size': 18, 'axes.unicode_minus': False, 'svg.fonttype': 'none'})
from matplotlib import pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import ARDRegression
import numpy as np
from random import shuffle


def convergence_datapts(X, y, sizes, model):
    num = X.shape[0]

    all_coeffs = {}
    num_fits = 0
    for nr in range(10, num, 20):
        num_fits += 1
        model.fit(X[:nr, :], y[:nr])
        coeff = model.coef_
        nonzero = list(np.nonzero(coeff)[0])

        for idx in nonzero:
            if idx in all_coeffs:
                all_coeffs[idx].append(coeff[idx])
            else:
                all_coeffs[idx] = [coeff[idx]]

    plot_occurence_hist(all_coeffs, num_fits, sizes)


def convergence_num_feat(X, y, sizes, model):
    num = X.shape[1]
    all_coeffs = {}
    num_fits = 0
    for nc in range(2, num):
        num_fits += 1
        model.fit(X[:, :nc], y)
        coeff = model.coef_
        nonzero = list(np.nonzero(coeff)[0])
        for idx in nonzero:
            if idx in all_coeffs:
                all_coeffs[idx].append(coeff[idx])
            else:
                all_coeffs[idx] = [coeff[idx]]
    frac_occ = {}
    for k, v in all_coeffs.items():
        denum = num - k if k > 2 else num - 1
        frac_occ[k] = len(v)/denum

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = [k for k in frac_occ.keys()]
    y = [frac_occ[k] for k in x]

    starts = [np.argmin(np.abs(sizes-2)),
              np.argmin(np.abs(sizes-3)),
              np.argmin(np.abs(sizes-4))]

    ax.bar(x[2:starts[1]], y[2:starts[1]], color='#766868', label='Pairs')
    ax.bar(x[starts[1]:starts[2]], y[starts[1]:starts[2]], color='#b63119',
           label='Triplets')
    ax.bar(x[starts[2]:], y[starts[2]:], color='#cb9f52', label='Quads')

    ax.set_xlabel("Feature no.")
    ax.set_ylabel("Fractional occurence")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(frameon=False)
    plt.show()


def lassoCVPath(X, y):
    model = LassoCV(cv=10, verbose=False, eps=1e-5)
    model.fit(X, y)
    fig_path = plt.figure()
    ax_path = fig_path.add_subplot(1, 1, 1)

    pred_error = np.mean(np.sqrt(model.mse_path_)*1000.0, axis=1)
    min_indx = np.argmin(pred_error)
    x_ax = np.log10(model.alphas_)

    ax_path.plot(x_ax, pred_error, color='#7c6868', label="CV")
    ax_path.axvline(np.log10(model.alphas_[min_indx]), ls='--', color='#7c6868')

    # Calculate AIC
    aicc_vals = np.zeros(len(model.alphas_))
    bic_vals = np.zeros_like(aicc_vals)
    nonzeros = []
    for i in range(len(model.alphas_)):
        m = Lasso(alpha=model.alphas_[i])
        m.fit(X, y)
        coeff = m.coef_
        nonzero = np.nonzero(coeff)[0]
        pred = m.predict(X)
        rss = np.sum((pred - y)**2)
        if rss < 1e-12:
            rss = 1e-12
        numCoeff = len(nonzero)
        nonzeros.append(nonzero)
        print(numCoeff, np.sqrt(rss/len(pred)), model.alphas_[i])
        aicc_vals[i] = aicc(numCoeff, X.shape[0], rss)
        bic_vals[i] = bic(numCoeff, X.shape[0], rss)

    ax_path2 = ax_path.twinx()
    ax_path2.plot(x_ax, aicc_vals, color='#b63119', label="AICc")

    min_indx = np.argmin(aicc_vals)
    ax_path.axvline(x_ax[min_indx], ls='--', color='#b63119')

    ax_path2.plot(x_ax, bic_vals, color='#cb9f52', label="BIC")

    min_indx = np.argmin(bic_vals)
    ax_path.axvline(x_ax[min_indx], ls='--', color='#cb9f52')
    ax_path.legend(frameon=False)

    ax_path.set_ylabel("CV (meV/atom)")
    ax_path.set_xlabel("log \$\\lambda\$")
    ax_path2.legend(frameon=False)
    ax_path2.set_ylabel("AICc/BIC")
    #ax_path2.set_ylim([-30000, -16500])

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)

    for i, non in enumerate(nonzeros):
        x = [np.log10(model.alphas_[i]) for _ in range(len(non))]
        ax2.plot(x, non, ls='none', marker='o', mfc='none', color='#7c6868',
                 markersize=1.5)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlabel("log \$\\lambda\$")
    ax2.set_ylabel("Feature no.")
    plt.show()


def bic(numCoeff, numData, rss):
    return numCoeff*np.log(numData) + numData*np.log(rss)


def aicc(numCoeff, numData, rss):
    return 2*numCoeff + (2*numCoeff**2 + 2*numCoeff)/(numData - numCoeff - 1) + numData*np.log(rss)


def random_subset_sampling(num_pts, num_repeats, X, y):
    indices = list(range(0, X.shape[0]))

    model = LassoCV(cv=5, verbose=False, eps=1e-5)
    feat_hist = np.zeros(X.shape[1])
    feat_hist_avg = np.zeros(X.shape[1])
    aicc_cut = 0.001

    # Standardize data
    y = (y - np.mean(y))/np.std(y)
    for i in range(1, X.shape[1]):
        X[:, i] = (X[:, i] - np.mean(X[:, i]))/np.std(X[:, i])

    tot_num_avg = 0
    for i in range(num_repeats):
        print("Sample {} of {}".format(i, num_repeats))
        shuffle(indices)
        indx = np.array(indices[:num_pts])
        X_sel = X[indx, :]
        y_sel = y[indx]
        model.fit(X_sel, y_sel)
        nonzero = np.nonzero(model.coef_)[0]
        print("Num coeff. {}".format(len(nonzero)))
        feat_hist[nonzero] += 1

        aicc_vals = np.zeros(len(model.alphas_))
        nonzeros = []
        # print(model.alphas_)
        # print(model.alpha_)
        # exit()
        for j in range(len(model.alphas_)):
            m = Lasso(alpha=model.alphas_[j])
            m.fit(X_sel, y_sel)
            coeff = m.coef_
            nonzero = np.nonzero(coeff)[0]
            pred = m.predict(X_sel)
            rmse = np.sqrt(np.mean((y_sel-pred)**2))
            rss = np.sum((y_sel - pred)**2)
            if rmse**2 < 1e-12:
                rmse = 1e-6
            #print("RMSE: {:e}".format(rmse))
            numCoeff = len(nonzero)
            nonzeros.append(nonzero)

            if numCoeff >= X_sel.shape[0] - 1:
                aicc_vals[j] = 1e100
            else:
                aicc_vals[j] = aicc(numCoeff, X_sel.shape[0], rss)

        aicc_vals -= np.min(aicc_vals)
        w = np.exp(-aicc_vals)
        #print(w)
        contribute = np.nonzero(w > aicc_cut)[0]
        print(contribute)
        for idx in contribute:
            feat_hist_avg[nonzeros[idx]] += 1.0
            tot_num_avg += 1

    feat_hist_avg /= tot_num_avg
    feat_hist /= num_repeats

    fname = 'random_partion{}.csv'.format(num_pts)
    np.savetxt(fname, np.vstack((feat_hist_avg, feat_hist)).T, delimiter=',')
    print("Selection data written to {}".format(fname))
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(feat_hist_avg, label='Avg')
    ax.plot(feat_hist, label='Opt.')
    ax.legend()

    print("Std avg: {}".format(np.std(feat_hist_avg)))
    print("Std opt: {}".format(np.std(feat_hist)))
    plt.show()




def plot_occurence_hist(all_coeffs, num_fits, sizes):
    hist = np.zeros(max(k for k in all_coeffs.keys())+1)
    hist = np.zeros(404)
    for k, v in all_coeffs.items():
        hist[k] = len(v)

    hist /= num_fits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(0, len(hist))

    starts = [np.argmin(np.abs(sizes-2)),
              np.argmin(np.abs(sizes-3)),
              np.argmin(np.abs(sizes-4))]

    ax.bar(x[2:starts[1]], hist[2:starts[1]], color='#766868', label='Pairs')
    ax.bar(x[starts[1]:starts[2]], hist[starts[1]:starts[2]], color='#b63119',
           label='Triplets')
    ax.bar(x[starts[2]:], hist[starts[2]:], color='#cb9f52', label='Quads')

    ax.set_xlabel("Feature no.")
    ax.set_ylabel("Fractional occurence")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(frameon=False)
    plt.show()


def check_uniqueness(X, y):
    indices = list(range(0, X.shape[1]))
    shuffle(indices)
    indices = np.array(indices)
    # Put the columns of X in a random order
    X = X[:, indices]
    model = LassoCV(eps=1e-4)
    model.fit(X, y)
    nonzero = np.nonzero(model.coef_)[0]
    print(np.sort(indices[nonzero]))


def main():
    fname = "agpt_data.csv"
    data = np.loadtxt(fname, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    print("Num features. {}".format(X.shape[1]))

    with open(fname, 'r') as infile:
        header = infile.readline()
        header = header.replace('#', '')
        header = header.replace(' ', '')
        splitted = header.split(',')[:-1]
        sizes = [int(x[1]) for x in splitted]

    model = LassoCV(cv=10)
    #model = ARDRegression(alpha_1=0.0, alpha_2=0.0)
    #convergence_datapts(X, y, np.array(sizes), model)
    #random_subset_sampling(200, 10, X, y)
    #check_uniqueness(X, y)
    #convergence_num_feat(X, y, np.array(sizes), model)
    lassoCVPath(X, y)


if __name__ == '__main__':
    main()
