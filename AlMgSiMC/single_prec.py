from apal import CHGLSinglePrecipitate2D
from apal import Khachaturyan
import numpy as np


class CHGLPoly:
    A = -0.2433564472289972
    B = -0.2710628512285493
    C = 13.553142561427464
    A_eta = -38.62645630006827
    B_eta = 37.08139804806554
    C_eta = -82.02921701562747
    D_eta = 82.02921701562747


def main():
    C_al = np.array([[0.62639459, 0.41086487, 0.41086487, 0, 0, 0],
                     [0.41086487, 0.62639459, 0.41086487, 0, 0, 0],
                     [0.41086487, 0.41086487, 0.62639459, 0, 0, 0],
                     [0, 0, 0, 0.42750351, 0, 0],
                     [0, 0, 0, 0, 0.42750351, 0],
                     [0, 0, 0, 0, 0, 0.42750351]])

    C_al *= 1000.0  # Convert to MeV!

    # Misfit strain in 3D
    misfit = np.array([[0.0440222, 0.00029263, 0.0008603],
                       [0.00029263, -0.0281846,   0.00029263],
                       [0.0008603,   0.00029263,  0.0440222]])

    khach = Khachaturyan(elastic_tensor=C_al, misfit_strain=misfit)

    alpha = 11.29325410426026
    beta = np.zeros((2, 2))
    beta[0, 0] = 1.3213819655965453
    beta[1, 1] = 5.82273736373145

    M = 1.0
    L = 1.0
    dx = 0.5
    solver = CHGLSinglePrecipitate2D(
        chgl_poly=CHGLPoly(), M=M, L=L, alpha=alpha, beta=beta,
        strain_model=khach, dx=dx, eta_max=0.8,
        prefix='/work/sophus/pure_python_calc/square_')

    conc = np.zeros((512, 512))
    conc[100:300] = 1.0
    eta = np.zeros((512, 512))
    eta[100:300] = 0.8
    solver.set_initial_conditions(conc, eta)
    solver.solve(nsteps=1000, backup_rate=100, dt=0.01, threshold=0.01)

main()
        