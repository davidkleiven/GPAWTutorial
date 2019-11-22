from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, ARDRegression
from matplotlib import pyplot as plt
import numpy as np
from random import shuffle

def main():
    X = np.loadtxt("data/cf_matrix.csv", delimiter=",")
    y = np.loadtxt("data/e_dft.csv", delimiter=",")

    y -= np.mean(y)


    indices = list(range(len(y)))
    shuffle(indices)

    train_indices = indices[:120]
    test_indices = indices[120:]
    X_train = X[train_indices, :]
    y_train = y[train_indices]

    X_test = X[test_indices, :]
    y_test = y[test_indices]



    X_train = X[:]
    regressor = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, loss='ls', max_depth=2, verbose=1, validation_fraction=0.1)
    base_estimator = Lasso(alpha=1E-3)
    regressor = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=300, learning_rate=0.2)
    est = regressor.fit(X_train, y_train)

    mse_test = mean_squared_error(y_test, est.predict(X_test))
    mse_train_set = mean_squared_error(y_train, est.predict(X_train))
    mse_train = []
    for pred in regressor.staged_predict(X_test):
        mse_train.append(mean_squared_error(y_test, pred))

    #plt.plot(regressor.train_score_)
    plt.plot(np.sqrt(mse_train)*1000)
    plt.show()

    plt.plot(y_train, est.predict(X_train), 'o', mfc='none')
    plt.plot(y_test, est.predict(X_test), 'o', mfc='none')
    plt.show()
    print(np.sqrt(mse_test)*1000.0)
    print(np.sqrt(mse_train_set)*1000.0)

if __name__ == '__main__':
    main()