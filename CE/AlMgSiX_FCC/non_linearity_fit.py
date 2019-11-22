import sys
from sklearn.linear_model import LassoCV
import numpy as np
from random import shuffle


def main(num_repeat):
    X = np.loadtxt("data/cf_matrix.csv", delimiter=",")
    y = np.loadtxt("data/e_dft.csv", delimiter=",")

    num_repetitions = 100

    fname = "data/non_lin_fit{}.txt".format(num_repeat)
    outfile = open(fname, 'w')
    for repeat in range(num_repetitions):
        outfile.write("# Repetition {}\n".format(repeat))
        outfile.write("# Iteration, MSE, MSE validation\n")
        optimise(X.copy(), y.copy(), outfile, num_power=num_repeat)
    outfile.close()


def optimise(X_raw, y, outfile, num_power=1):
    if num_power == 1:
        X = X_raw
    elif num_power == 2:
        X = np.hstack((X_raw, X_raw**2))
    elif num_power == 3:
        X = np.hstack((X_raw, X_raw**2, X_raw**3))
    elif num_power == 4:
        X = np.hstack((X_raw, X_raw**2, X_raw**3, X_raw**4))
    elif num_power == 5:
        X = np.hstack((X_raw, X_raw**2, X_raw**3, X_raw**4, X_raw**5))
    elif num_power == 6:
        X = np.hstack((X_raw, X_raw**2, X_raw**3, X_raw**4, X_raw**5, X_raw**6))

    # Extract 20 structure as validation set
    indices = list(range(X.shape[0]))
    shuffle(indices)
    validation = indices[:20]

    X_val = X[validation, :]
    y_val = y[validation]

    trainX = np.delete(X, validation, axis=0)
    trainy = np.delete(y, validation)
    print(trainX.shape, X.shape)

    assert trainX.shape[0] == 128
    assert len(trainy) == 128

    model = LassoCV(eps=1E-3, n_alphas=100, cv=10)
    reg = model.fit(trainX, trainy)
    coeff = model.coef_.copy()
    pred = model.predict(trainX)
    pred_val = model.predict(X_val)

    mse = np.sqrt(np.mean((pred - trainy)**2))*1000.0
    val_mse = np.sqrt((np.mean((pred_val - y_val)**2)))*1000.0
    outfile.write('{},{},{}\n'.format(0, mse, val_mse))
    outfile.flush()

    for num in range(1, 100):
        y2 = trainy - pred
        model.fit(trainX, y2)
        pred += model.predict(trainX)
        pred_val += model.predict(X_val)
        mse = np.sqrt(np.mean((pred - trainy)**2))*1000.0
        val_mse = np.sqrt((np.mean((pred_val - y_val)**2)))*1000.0

        if np.allclose(model.coef_, 0.0):
            break
        outfile.write('{},{},{}\n'.format(num, mse, val_mse))
        outfile.flush()

if __name__ == "__main__":
    main(int(sys.argv[1]))