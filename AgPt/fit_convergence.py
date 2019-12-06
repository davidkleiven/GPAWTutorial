import matplotlib as mpl
mpl.rcParams.update({'font.size': 18, 'axes.unicode_minus': False, 'svg.fonttype': 'none'})
from matplotlib import pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
import numpy as np


def convergence_datapts(X, y, sizes):
    num = X.shape[0]

    model = LassoCV(cv=10)

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


def convergence_num_feat(X, y, sizes):
    num = X.shape[1]
    all_coeffs = {}

    model = LassoCV(cv=10, verbose=False)
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

    logpred = np.log10(pred_error)
    rng = np.max(logpred) - np.min(logpred)
    logpred /= rng
    ax_path.plot(x_ax, logpred, color='#7c6868', label="CV")
    ax_path.axvline(np.log10(model.alphas_[min_indx]), ls='--', color='#7c6868')

    # Calculate AIC
    aicc_vals = np.zeros(len(model.alphas_))
    bic_vals = np.zeros_like(aicc_vals)
    for i in range(len(model.alphas_)):
        m = Lasso(alpha=model.alphas_[i])
        m.fit(X, y)
        coeff = m.coef_
        nonzero = np.nonzero(coeff)[0]
        pred = m.predict(X)
        rmse = np.sqrt(np.mean((y-pred)**2))
        numCoeff = len(nonzero)
        print(numCoeff, rmse, model.alphas_[i])
        aicc_vals[i] = aicc(numCoeff, X.shape[0], rmse)
        bic_vals[i] = bic(numCoeff, X.shape[0], rmse)

    mean_log_pred = np.mean(np.log10(pred_error))
    aicc_vals += mean_log_pred - np.mean(aicc_vals)
    rng = np.max(aicc_vals) - np.min(aicc_vals)
    ax_path.plot(x_ax, aicc_vals/rng, color='#b63119', label="AICc")

    min_indx = np.argmin(aicc_vals)
    ax_path.axvline(x_ax[min_indx], ls='--', color='#b63119')

    bic_vals += mean_log_pred - np.mean(bic_vals)
    rng = np.max(bic_vals) - np.min(bic_vals)
    ax_path.plot(x_ax, bic_vals/rng + 0.2, color='#cb9f52', label="BIC")

    min_indx = np.argmin(bic_vals)
    ax_path.axvline(x_ax[min_indx], ls='--', color='#cb9f52')
    ax_path.legend(frameon=False)

    ax_path.set_ylabel("Normalised score")
    ax_path.set_xlabel("log \$\\lambda\$")
    ax_path.spines['right'].set_visible(False)
    ax_path.spines['top'].set_visible(False)
    plt.show()


def bic(numCoeff, numData, rmse):
    return numCoeff*np.log(numData) + 2*numData*np.log(rmse)


def aicc(numCoeff, numData, rmse):
    return 2*numCoeff + (2*numCoeff**2 + 2*numCoeff)/(numData - numCoeff - 1) + 2*numData*np.log(rmse)


def plot_occurence_hist(all_coeffs, num_fits, sizes):
    hist = np.zeros(max(k for k in all_coeffs.keys())+1)
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


def main():
    fname = "agpt_data.csv"
    data = np.loadtxt(fname, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]

    with open(fname, 'r') as infile:
        header = infile.readline()
        header = header.replace('#', '')
        header = header.replace(' ', '')
        splitted = header.split(',')[:-1]
        sizes = [int(x[1]) for x in splitted]
    convergence_datapts(X, y, np.array(sizes))
    #convergence_num_feat(X, y, np.array(sizes))
    # lassoCVPath(X, y)


if __name__ == '__main__':
    main()
