from matplotlib import pyplot as plt
import json
import numpy as np

FOLDER = "/work/sophus/nestedlasso/"


def get_coeff_aic(fname):
    with open(fname, 'r') as infile:
        data = json.load(infile)

    s = []
    c = []
    aicc = []

    for item in data['Paths']:
        for i, node in enumerate(item['Nodes']):
            if node['Selection'] not in s:
                s.append(node['Selection'])
                c.append(node['Coeff'])
                aicc.append(item['Aicc'][i])
    return s, c, aicc


def get_coeff_stat_best(all_s, all_c, all_aicc):
    coeff = {}

    for s, c, a in zip(all_s, all_c, all_aicc):
        best = np.argmin(a)
        for i in range(len(s[best])):
            if s[best][i] not in coeff.keys():
                coeff[s[best][i]] = [c[best][i]]
            else:
                coeff[s[best][i]].append(c[best][i])

    coeff_stat = {}

    for k, v in coeff.items():
        coeff_stat[k] = {
            'mean': np.mean(v),
            'std': np.std(v),
            'num': len(v)
        }
    return coeff_stat


def get_coeff_stat_aic(all_s, all_c, all_aicc):
    coeff = {}
    for s, c, a in zip(all_s, all_c, all_aicc):
        a = np.array(a) - np.min(a)
        expa = np.exp(-a)
        indices = np.nonzero(expa > 0.01)[0]
        print(len(indices))

        w = expa/np.sum(expa[expa > 0.01])

        tmp_coeff = {}
        for idx in indices:
            for i in range(len(s[idx])):
                if s[idx][i] not in tmp_coeff.keys():
                    tmp_coeff[s[idx][i]] = w[idx]*c[idx][i]
                else:
                    tmp_coeff[s[idx][i]] += c[idx][i]

        for k, v in tmp_coeff.items():
            if k not in coeff.keys():
                coeff[k] = [v]
            else:
                coeff[k].append(v)

    coeff_stat = {}
    for k, v in coeff.items():
        coeff_stat[k] = {
            'mean': np.mean(v),
            'std': np.std(v),
            'num': len(v)
        }
    return coeff_stat


def main():
    all_c = []
    all_s = []
    all_a = []
    for i in range(100):
        fname = FOLDER + "/nestedlasso{}.json".format(i)
        s, c, a = get_coeff_aic(fname)
        all_s.append(s)
        all_c.append(c)
        all_a.append(a)

    coeff_best = get_coeff_stat_best(all_s, all_c, all_a)
    coeff_aic = get_coeff_stat_aic(all_s, all_c, all_a)

    n = 404
    num_aic = np.zeros(n)
    num_b = np.zeros(n)

    for k, v in coeff_best.items():
        num_b[k] = v['num']

    for k, v in coeff_aic.items():
        num_aic[k] = v['num']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(num_b)
    ax.plot(num_aic)
    plt.show()


main()

