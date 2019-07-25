#from cemc.phasefield import PyCHGL
import sys
from apal_cxx import PyCHGL
from apal_cxx import PyCHGLRealSpace
from apal_cxx import PyKernelRegressor
from apal_cxx import PyGaussianKernel
from apal_cxx import PyPolynomial
from apal_cxx import PyTwoPhaseLandau
from apal_cxx import PyQuadraticTwoPhasePoly
from apal_cxx import PyPolynomialTerm
from apal_cxx import PyKhachaturyan
from matplotlib import pyplot as plt
from cemc.tools import to_full_rank4
import numpy as np
#from cemc.phasefield.tools import get_polyterms
from apal.tools import get_polyterms
import json

FNAME = "chgl_almgsi_apal.json"
FNAME = "chgl_almgsi_apal_non_zero_grad_coeff.json"
FNAME = "chgl_almgsi_apal_non_zero_grad_coeff_quadratic.json"
FNAME = "chgl_almgsi_quadratic_large_alpha.json"

khac1 = None
khac2 = None
def add_strain(chgl):
    global khac1, khac2

    C_al = np.array([[0.62639459, 0.41086487, 0.41086487, 0, 0, 0],
            [0.41086487, 0.62639459, 0.41086487, 0, 0, 0],
            [0.41086487, 0.41086487, 0.62639459, 0, 0, 0],
            [0, 0, 0, 0.42750351, 0, 0],
            [0, 0, 0, 0, 0.42750351, 0],
            [0, 0, 0, 0, 0, 0.42750351]])
    
    C_al *= 1000.0 # Convert to MeV!
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


    khac1 = PyKhachaturyan(2, C_al, misfit1)
    khac2 = PyKhachaturyan(2, C_al, misfit2)

    chgl.add_strain_model(khac1, 1)
    chgl.add_strain_model(khac2, 2)


def main(prefix, start, startfile, initfunc, dx=30.0, steps=0, update_freq=0):
    #prefix = "data/almgsi_chgl_random_seed_strain_noise2/chgl"
    #prefix = "data/almgsi_chgl_3D_surface_3nm_64_strain_meV/chgl"
    #prefix = "data/almgsi_chgl_3D_MLdx1_1nm_64_strain_meV/chgl"
    #prefix = "/work/sophus/almgsi_chgl_3D_MLdx1_3nm_64_strain_meV/chgl"
    dim = 2
    L = 128
    num_gl_fields = 2
    M = 0.1/dx**2
    alpha = 5.0
    dt = 0.003
    gl_damping = M/dx**2

    coeff, terms = get_polyterms(FNAME)

    poly = PyPolynomial(3)
    poly1 = PyPolynomial(1)

    with open(FNAME, 'r') as infile:
        info = json.load(infile)
    
    # kernel = PyGaussianKernel(info["kernel"]["std_dev"])
    # regressor = PyKernelRegressor(
    #     info["kernel_regressor"]["xmin"],
    #     info["kernel_regressor"]["xmax"])
    # regressor.set_kernel(kernel)
    # regressor.set_coeff(info["kernel_regressor"]["coeff"])
    grad_coeff = info["gradient_coeff"]

    for item in info["terms"]:
        c = item["coeff"]
        powers = item["powers"]
        if powers[-1] > 0:
            continue
        poly.add_term(c, PyPolynomialTerm(powers[:-1]))
        print(c, powers)

    N = len(info["conc_phase1"])
    for i, c in enumerate(info["conc_phase1"]):
        poly1.add_term(c, PyPolynomialTerm([N-i-1]))

    alpha = grad_coeff[0]/dx**2
    b1 = grad_coeff[1]/dx**2
    b2 = grad_coeff[2]/dx**2
    gradient_coeff = [[b2, b1],
                      [b1, b2]]
    print(gradient_coeff)

    chgl = PyCHGLRealSpace(dim, L, prefix, num_gl_fields, M, alpha, dt,
                           gl_damping, gradient_coeff)

    # landau = PyTwoPhaseLandau()
    # landau.set_polynomial(poly)
    # landau.set_kernel_regressor(regressor)
    # landau.set_discontinuity(info["discontinuity_conc"], info["discontinuity_jump"])
    landau = PyQuadraticTwoPhasePoly()
    landau.set_poly_phase1(poly1)
    landau.set_poly_phase2(poly)

    chgl.set_free_energy_quadratic(landau)
    chgl.use_adaptive_stepping(1E-10, 1, 0.005)
    chgl.set_field_update_rate(10)
    chgl.set_strain_update_rate(100)
    chgl.build2D()
    add_strain(chgl)

    if startfile is not None:
        chgl.from_file(prefix + startfile)
    else:
        if initfunc == "precipitate_square":
            chgl.from_npy_array(precipitates_square(L))
        elif initfunc == "matsuda":
            chgl.from_npy_array(create_matsuda(L))
        elif initfunc == 'prec_square_bck':
            chgl.from_npy_array(precipitate_square_bck(L))
        else:
            raise ValueError("Unknown init function!")
            

    chgl.run(steps, update_freq, start=start)
    chgl.save_free_energy_map(prefix+"_free_energy_map.grid")


def precipitates_square(L):
    conc = np.zeros((L, L))
    start = 20
    end = 80
    conc[start:end, start:end] = 1.0

    eta1 = np.zeros((L, L))
    eta1[start:end, start:end, start:end] = 0.8
    eta2 = np.zeros((L, L))

    from scipy.ndimage import gaussian_filter
    conc = gaussian_filter(conc, 3.0)
    eta1 = gaussian_filter(eta1, 3.0)
    return [conc, eta1, eta2]


def create_matsuda(L):
    conc = np.zeros((L, L, L))
    conc[19:29, 16:48, 16:48] = 1.0
    conc[35:45, 16:48, 16:48] = 1.0
    eta1 = np.zeros((L, L, L))
    eta1[19:29, 16:48, 16:48] = 0.8
    eta1[35:45, 16:48, 16:48] = 0.8
    eta2 = np.zeros((L, L, L))
    eta3 = np.zeros((L, L, L))
    return [conc, eta1, eta2, eta3]


def precipitate_square_bck(L):
    conc = np.random.rand(L, L, L)*0.4
    prec = precipitates_square(L)
    mask = prec[0] < 0.05
    prec[0][mask] = conc[mask]
    return prec


if __name__ == "__main__":
    prefix = None
    start = 0
    startfile = None
    initfunc = None
    dx = 30.0
    num_steps = 0
    update_freq = 0

    for argv in sys.argv:
        if "--help" in argv:
            print("Usage: --prefix=<folder to store grid file>")
            print("--start=<start iteration>")
            print("--startfile=<file to start from> (overall filename is prefix+startfile")
            print("--initfunc=<initialisation function>")
            print("--dx=<grid discretisation> in Angstrom") 
            print("--steps=<Number of simulation steps>")
            print("--update_freq=<Save results every>")
            exit(0)

    for argv in sys.argv[1:]:
        if '--prefix=' in argv:
            prefix = argv.split('--prefix=')[1]
        elif '--startiter=' in argv:
            start = int(argv.split('--startiter=')[1])
        elif '--startfile=' in argv:
            startfile = argv.split('--startfile=')[1]
        elif '--initfunc=' in argv:
            initfunc = argv.split('--initfunc=')[1]
        elif '--dx=' in argv:
            dx = float(argv.split('--dx=')[1])
        elif '--steps=' in argv:
            num_steps = int(argv.split('--steps=')[1])
        elif '--update_freq=' in argv:
            update_freq = int(argv.split('--update_freq=')[1])
        else:
            raise ValueError("Unknown option {}".format(argv))

    assert prefix is not None
    assert num_steps > 0
    assert update_freq > 0
    if startfile is not None:
        assert str(start) in startfile
        assert initfunc is None
    else:
        assert initfunc in ['matsuda', 'precipitate_square', 'prec_square_bck']

    main(prefix, start, startfile, initfunc, dx, num_steps, update_freq)
