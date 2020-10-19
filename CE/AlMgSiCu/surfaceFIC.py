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
	variances = var*(1.0 - 2.0/len(dy)) + np.sum(var)/len(dy)**2
	var_slope = np.sum(dx**2*variances)/s
	return slope, var_slope

def fic_single(X, y, coeff, names, target_data, verbose=False):
	pred = X.dot(coeff)
	prec = np.linalg.inv(X.T.dot(X))
	rmse = np.sqrt(np.mean((pred - y)**2))
	rows, cols = X.shape
	if rows > cols:
		rmse *= np.sqrt(rows/(rows - cols))
	print(f"RMSE: {rmse}")

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

	# Try to fit the energies
	bias = np.mean((E_dft - pred)**2)

	if verbose:
		print(f"Inv. size: {inv_size}")
		print(f"Slope DFT: {slope_dft}")
		print(f"Slope CE: {slope_ce} std {np.sqrt(slope_var)}")

		from matplotlib import pyplot as plt
		plt.plot(inv_size, E_dft, 'x')
		plt.plot(inv_size, pred, 'o')
		plt.show()
	return np.sqrt(bias + np.mean(var))
	#return np.sqrt(bias**2 + slope_var)

def fic(X, y, coeff, names):
	fic1 = fic_single(X, y, coeff, names, surfaceAlt)
	fic2 = fic_single(X, y, coeff, names, surfacePure)
	return fic1 + fic2

def from_model(model_file):
	with open(model_file, 'r') as infile:
		data = json.load(infile)
		coeff = data["Coeffs"]
	
	data = np.loadtxt("data/almgsicu.csv", skiprows=1, delimiter=",")
	X = data[:, :-1]
	y = data[:, -1]
	with open("data/almgsicu.csv", 'r') as infile:
		header = infile.readline().strip()
		header = header.replace("#", "")
		header = header.replace(" ", "")
		header = header.split(",")
	cols = [header.index(k) for k in coeff.keys()]
	vals = np.array(list(coeff.values()))
	X = X[:, cols]
	print(f"No. coeffs: {len(coeff)}")
	fic_single(X, y, vals, list(coeff.keys()), surfaceAlt, verbose=True)
	fic_single(X, y, vals, list(coeff.keys()), surfacePure, verbose=True)


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

#from_model("data/ga_surface_fic.json")
#from_model("data/ga_almgsicu_aicc.json")
main(sys.argv[1])
