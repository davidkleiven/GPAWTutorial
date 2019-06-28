#from cemc.phasefield import PyCHGL
from apal_cxx import PyCHGL
from apal_cxx import PyCHGLRealSpace
from apal_cxx import PyKernelRegressor
from apal_cxx import PyGaussianKernel
from apal_cxx import PyPolynomial
from apal_cxx import PyTwoPhaseLandau
from apal_cxx import PyPolynomialTerm
from apal_cxx import PyKhachaturyan
from matplotlib import pyplot as plt
from cemc.tools import to_full_rank4
import numpy as np
from cemc.phasefield.tools import get_polyterms
import json

FNAME = "chgl_almgsi_apal.json"

khac1 = None
khac2 = None
khac3 = None
def add_strain(chgl):
    global khac1, khac2, khac3

    C_al = np.array([[0.62639459, 0.41086487, 0.41086487, 0, 0, 0],
            [0.41086487, 0.62639459, 0.41086487, 0, 0, 0],
            [0.41086487, 0.41086487, 0.62639459, 0, 0, 0],
            [0, 0, 0, 0.42750351, 0, 0],
            [0, 0, 0, 0, 0.42750351, 0],
            [0, 0, 0, 0, 0, 0.42750351]])
    
    C_al = to_full_rank4(C_al)
    # Misfit strain in 3D
    misfit = np.array([[ 0.0440222,   0.00029263,  0.0008603 ],
                        [ 0.00029263, -0.0281846,   0.00029263],
                        [ 0.0008603,   0.00029263,  0.0440222 ]])

    misfit1 = misfit = np.array([[ 0.0440222,   0.00029263,  0.0 ],
                        [ 0.00029263, -0.0281846,   0.0],
                        [ 0.0,   0.0,  0.0440222 ]])


    misfit2 = np.array([[ -0.0281846,   0.00029263,  0.0 ],
                        [ 0.00029263, 0.0440222,   0.0],
                        [ 0.0,   0.0,  0.0440222 ]])

    misfit3 = np.array([[ 0.0440222,   0.00029263,  0.0 ],
                        [ 0.00029263, 0.0440222,   0.0],
                        [ 0.0,   0.0,  -0.0281846 ]])

    khac1 = PyKhachaturyan(3, C_al, misfit1)
    khac2 = PyKhachaturyan(3, C_al, misfit2)
    khac3 = PyKhachaturyan(3, C_al, misfit3)

    chgl.add_strain_model(khac1, 1)
    chgl.add_strain_model(khac2, 2)
    chgl.add_strain_model(khac3, 3)


def main():
    #prefix = "data/almgsi_chgl_random_seed_strain_noise2/chgl"
    prefix = "data/almgsi_chgl_3D_surface_1nm_64_strain_consistent/chgl"
    dx = 10.0  # Discretisation in angstrom
    dim = 3
    L = 64
    num_gl_fields = 3
    M = 0.1
    alpha = 5.0
    dt = 1.0
    gl_damping = M

    coeff, terms = get_polyterms(FNAME)

    poly = PyPolynomial(4)

    with open(FNAME, 'r') as infile:
        info = json.load(infile)
    
    kernel = PyGaussianKernel(info["kernel"]["std_dev"])
    regressor = PyKernelRegressor(
        info["kernel_regressor"]["xmin"],
        info["kernel_regressor"]["xmax"])
    regressor.set_kernel(kernel)
    regressor.set_coeff(info["kernel_regressor"]["coeff"])
    grad_coeff = info["gradient_coeff"]

    for item in info["terms"]:
        c = item["coeff"]
        powers = item["powers"]
        poly.add_term(c, PyPolynomialTerm(powers))
        print(c, powers)

    alpha = grad_coeff[0]/dx**2
    b1 = grad_coeff[1]/dx**2
    b2 = grad_coeff[2]/dx**2
    gradient_coeff = [[b2, b1, b2],
                      [b1, b2, b2],
                      [b2, b2, b1]]
    print(gradient_coeff)

    chgl = PyCHGLRealSpace(dim, L, prefix, num_gl_fields, M, alpha, dt,
                           gl_damping, gradient_coeff)
    
    landau = PyTwoPhaseLandau()
    landau.set_polynomial(poly)
    landau.set_kernel_regressor(regressor)
    landau.set_discontinuity(info["discontinuity_conc"], info["discontinuity_jump"])

    chgl.set_free_energy(landau)
    #chgl.from_npy_array(precipitates_square(L))
    chgl.use_adaptive_stepping(1E-10, 1, 0.05)
    chgl.build3D()
    add_strain(chgl)
    chgl.from_file(prefix + "00000053000.grid")

    chgl.run(500000, 5000, start=53000)
    chgl.save_free_energy_map(prefix+"_free_energy_map.grid")

def precipitates_square(L):
    conc = np.zeros((L, L, L))
    conc[22:42, 22:42, 22:42] = 1.0

    eta1 = np.zeros((L, L, L))
    eta1[22:42, 22:42, 22:42] = 0.8
    eta2 = np.zeros((L, L, L))
    eta3 = np.zeros((L, L, L))
    return [conc, eta1, eta2, eta3]

if __name__ == "__main__":
    main()
