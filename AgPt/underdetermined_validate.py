import numpy as np
import json

FOLDER = "/work/sophus/nestedlasso"
NUM = 100
data_val = np.loadtxt("agpt_validate.csv", delimiter=',')
X_val = data_val[:, :-1]
y_val = data_val[:, -1]


def validate(selection, coeff):
    pred = X_val[:, selection].dot(coeff)
    mse = np.sqrt(np.mean((pred - y_val)**2))
    return mse


def predErrorThresholdLasso():
    for i in range(NUM):
        fname = FOLDER + "/thresholdLasso{}.json".format(i)
        #fname = FOLDER + "/empCovLasso{}.json".format(i)
        with open(fname, 'r') as infile:
            data = json.load(infile)

        aicc = data['Aicc']
        idx = np.argmin(aicc)
        coeff = data['LassoLarsNodes'][idx]['Coeff']
        selection = data['LassoLarsNodes'][idx]['Selection']

        # X = data['Dset']['X']
        # y = data['Dset']['Y']
        # X = np.reshape(X, (200, 403))
        # coeff, _, _, _ = np.linalg.lstsq(X[:, selection], y)

        mse = validate(selection, coeff)*1000.0
        print("Validation error {} meV/atom. Num coeff. {}"
              "".format(mse, len(selection)))


predErrorThresholdLasso()
