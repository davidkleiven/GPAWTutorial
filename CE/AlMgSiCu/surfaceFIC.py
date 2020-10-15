#!/usr/bin/env python3
import sys
import json
import numpy as np

surfaceAlt= "data/surfaceAlt.csv"
surfacePure = "data/surfacePure.csv"

def slope(x, y, var=None):
	xbar = np.mean(x)
	ybar = np.mean(y)
	dx = x - xbar
	dy = y - ybar
	slope = np.sum(dx*dy)/np.sum(dx**2)

	if var is None:
		return slope, None
	
	s = np.sum(dx**2)**2
	var_slope = np.sum(dx**2*var)/s
	return slope, var_slope

def fic_single(X, y, coeff, names, target_data):
	pred = X.dot(coeff)
	prec = np.linalg.inv(X.T.dot(X))
	rmse = np.sqrt(np.mean(pred - y)**2)
	rows, cols = X.shape
	if rows > cols:
		rmse *= np.sqrt(rows/(rows - cols))

	data = np.loadtxt(target_data, skiprows=1, delimiter=",")
	X = data[:, :-2]
	E_dft = data[:, -2]
	size = data[:,-1]
	E_dft /= size
	inv_size = 1.0/size
	with open(target_data, 'r') as infile:
		header = infile.readline().strip()
		header = header.split(",")
	
	header_col = {k: i for i, k in enumerate(header)}
	cols = [header_col[k] for k in names]
	X = X[:, cols]
	pred = X.dot(coeff)

	slope_dft, _ = slope(inv_size, E_dft)
	var = rmse**2*(1.0 + np.diag(X.dot(prec).dot(X.T)))
	slope_ce, slope_var = slope(inv_size, pred, var=var)
	bias = slope_ce - slope_dft
	return np.sqrt(bias**2 + slope_var)

def fic(X, y, coeff, names):
	fic1 = fic_single(X, y, coeff, names, surfaceAlt)
	fic2 = fic_single(X, y, coeff, names, surfacePure)
	return fic1 + fic2


def main(arg):
	with open(arg, 'r') as infile:
		args = json.load(infile)

	# When gogafit calls this script, args will now contain
	# {
	#     "Rows": <number of rows in X>
	#     "Cols": <number of columns in X>
	#     "X": <Design matrix>
	#     "Y": <Target value>
	#     "Coeff": <Fitted coefficients>
	# }
	# Predictions for Y can be obtained via y_pred = X.dot(Coeff)

	X = np.reshape(args["X"], (args["Rows"], args["Cols"]))
	y = np.array(args["Y"])
	coeff = np.array(args["Coeff"])
	names = args["Names"]

    # # Do your calculations, and store the result in this variable
	cost_value = fic(X, y, coeff, names)

    # Important: The following print statement must be present
    # if gogafit should be able to extract
	print("GOGAFIT_COST: {}".format(cost_value))

main(sys.argv[1])
	