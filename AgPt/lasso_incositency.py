from sklearn.linear_model import LassoCV
import numpy as np
from random import shuffle


def fit_reordered(X, y):
    indices = list(range(0, X.shape[1]))
    shuffle(indices)
    indices = np.array(indices)

    # Order the columns of X in a random order
    X = X[:, indices]
    model = LassoCV(eps=1e-3)
    model.fit(X, y)
    coef = model.coef_
    nonzero = np.nonzero(coef)[0]
    print(np.sort(indices[nonzero]))


def fit_cyclic_vs_random(X, y):
    model_cyc = LassoCV(eps=1e-3, selection='cyclic')
    model_rnd = LassoCV(eps=1e-3, selection='random')
    model_cyc.fit(X, y)
    model_rnd.fit(X, y)

    nonzero_cyc = np.nonzero(model_cyc.coef_)[0]
    nonzero_rnd = np.nonzero(model_rnd.coef_)[0]
    print(nonzero_cyc)
    print()
    print(nonzero_rnd)


data = np.loadtxt("agpt_data_compressed.csv", delimiter=',')
X = data[:, :-1]
y = data[:, -1]
#fit_cyclic_vs_random(X, y)
fit_reordered(X, y)
