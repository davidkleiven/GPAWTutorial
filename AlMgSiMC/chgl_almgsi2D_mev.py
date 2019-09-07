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
from scipy.ndimage import gaussian_filter

FNAME = "chgl_almgsi_apal.json"
FNAME = "chgl_almgsi_apal_non_zero_grad_coeff.json"
FNAME = "chgl_almgsi_apal_non_zero_grad_coeff_quadratic.json"
FNAME = "chgl_almgsi_quadratic_large_alpha.json"

FNAME = "chgl_almgsi_quadratic_large_alpha600K.json"
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

    misfit1 = np.array([[ 0.0440222,   0.00029263,  0.0 ],
                        [ 0.00029263, -0.0281846,   0.0],
                        [ 0.0,   0.0,  0.0440222 ]])


    misfit2 = np.array([[ -0.0281846,   0.00029263,  0.0 ],
                        [ 0.00029263, 0.0440222,   0.0],
                        [ 0.0,   0.0,  0.0440222 ]])


    khac1 = PyKhachaturyan(2, C_al, misfit1)
    khac2 = PyKhachaturyan(2, C_al, misfit2)

    chgl.add_strain_model(khac1, 1)
    chgl.add_strain_model(khac2, 2)


def main(prefix, start, startfile, initfunc, dx=30.0, dt=0.3, steps=0, update_freq=0, prec_x0=20, prec_x1=50, a=1, b=1):
    #prefix = "data/almgsi_chgl_random_seed_strain_noise2/chgl"
    #prefix = "data/almgsi_chgl_3D_surface_3nm_64_strain_meV/chgl"
    #prefix = "data/almgsi_chgl_3D_MLdx1_1nm_64_strain_meV/chgl"
    #prefix = "/work/sophus/almgsi_chgl_3D_MLdx1_3nm_64_strain_meV/chgl"
    dim = 2
    L = 512
    num_gl_fields = 2
    M = 0.1/dx**2
    alpha = 5.0
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
    #gradient_coeff = [[b2, b1],
    #                  [b1, b2]]
    gradient_coeff = [[b1, b2],
                      [b2, b1]]

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
    chgl.use_adaptive_stepping(1E-10, 1, 0.05)
    #chgl.use_HeLiuTang_stabilizer(500)
    #chgl.set_field_update_rate(100)
    #chgl.set_strain_update_rate(1000)
    chgl.build2D()
    add_strain(chgl)
    chgl.conserve_volume(1)
    chgl.conserve_volume(2)

    if startfile is not None:
        chgl.from_file(prefix + startfile)
    else:
        if initfunc == "precipitate_square":
            chgl.from_npy_array(precipitates_square(L, start=prec_x0, end=prec_x1))
        elif initfunc == "matsuda":
            chgl.from_npy_array(create_matsuda(L))
        elif initfunc == 'prec_square_bck':
            chgl.from_npy_array(precipitate_square_bck(L))
        elif initfunc == 'random':
            chgl.from_npy_array(random_orientation(L))
        elif initfunc == 'ellipse':
            chgl.from_npy_array(ellipse(L, a, b))
        else:
            raise ValueError("Unknown init function!")
            

    chgl.run(steps, update_freq, start=start)
    chgl.save_free_energy_map(prefix+"_free_energy_map.grid")


def precipitates_square(L, start=20, end=50):
    conc = np.zeros((L, L))
    conc[start:end, start:end] = 1.0

    eta1 = np.zeros((L, L))
    eta2 = np.zeros((L, L))
    eta2[start:end, start:end] = 0.8

    from scipy.ndimage import gaussian_filter
    conc = gaussian_filter(conc, 10.0)
    eta2 = gaussian_filter(eta2, 10.0)
    return [conc, eta1, eta2]


def create_matsuda(L):
    conc = np.zeros((L, L))
    conc[20:40, 20:100] = 1.0
    conc[60:80, 20:100] = 1.0
    eta1 = np.zeros((L, L))
    eta1[20:40, 20:100] = 0.8
    eta1[60:80, 20:100] = 0.8
    eta2 = np.zeros((L, L))
    eta3 = np.zeros((L, L))
    return [conc, eta1, eta2]


def precipitate_square_bck(L):
    conc = np.random.rand(L, L, L)*0.4
    prec = precipitates_square(L)
    mask = prec[0] < 0.05
    prec[0][mask] = conc[mask]
    return prec

def random_orientation(L):
    num_prec = 200
    size = 10

    conc = np.zeros((L, L))
    eta1 = np.zeros((L, L))
    eta2 = np.zeros((L, L))

    num_inserted = 0
    for i in range(10000):
        x0 = np.random.randint(size, high=L-size)
        y0 = np.random.randint(size, high=L-size)

        if np.any(conc[x0:x0+size, y0:y0+size] > 0.2):
            continue

        num_inserted += 1

        conc[x0:x0+size, y0:y0+size] = 1.0

        if np.random.rand() < 0.5:
            eta1[x0:x0+size, y0:y0+size] = 1.0
        else:
            eta2[x0:x0+size, y0:y0+size] = 1.0

        if num_inserted >= num_prec:
            break

    conc = gaussian_filter(conc, 4)
    eta1 = gaussian_filter(eta1, 4)
    eta2 = gaussian_filter(eta2, 4)
    return [conc, eta1, eta2]


def ellipse(L, a, b):
    conc = np.zeros((L, L))
    eta1 = np.zeros((L, L))
    eta2 = np.zeros((L, L))

    c = L/2
    x = np.linspace(-c, c, L)
    X, Y = np.meshgrid(x, x)
    ellipse_eq = (X/a)**2 + (Y/b)**2
    mask = ellipse_eq <= 1.0
    conc[mask] = 1.0
    eta2[mask] = 0.8
    conc = gaussian_filter(conc, 8)
    eta1 = gaussian_filter(eta1, 8)
    eta2 = gaussian_filter(eta2, 8)
    return [conc, eta1, eta2]

if __name__ == "__main__":
    prefix = None
    start = 0
    startfile = None
    initfunc = None
    dx = 30.0
    num_steps = 0
    update_freq = 0
    prec_x0 = 20
    prec_x1 = 50
    a = 1
    b = 1
    dt = 0.3

    for argv in sys.argv:
        if "--help" in argv:
            print("Usage: --prefix=<folder to store grid file>")
            print("--start=<start iteration>")
            print("--startfile=<file to start from> (overall filename is prefix+startfile")
            print("--initfunc=<initialisation function>")
            print("--dx=<grid discretisation> in Angstrom") 
            print("--steps=<Number of simulation steps>")
            print("--update_freq=<Save results every>")
            print("--prex_x0=<Start square precipitate (default 20)>")
            print("--prex_x1=<End square precipitate (default 50)>")
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
        elif '--prec_x0=' in argv:
            prec_x0 = int(argv.split('--prec_x0=')[1])
        elif '--prec_x1=' in argv:
            prec_x1 = int(argv.split('--prec_x1=')[1])
        elif '--a=' in argv:
            a = int(argv.split('--a=')[1])
        elif '--b=' in argv:
            b = int(argv.split('--b=')[1])
        elif '--dt=' in argv:
            dt = float(argv.split('--dt=')[1])
        else:
            raise ValueError("Unknown option {}".format(argv))

    assert prefix is not None
    assert num_steps > 0
    assert update_freq > 0
    if startfile is not None:
        assert str(start) in startfile
        assert initfunc is None
    else:
        assert initfunc in ['matsuda', 'precipitate_square', 'prec_square_bck', 'random', 'ellipse']

    main(prefix, start, startfile, initfunc, dx, dt, num_steps, update_freq, prec_x0, prec_x1, a, b)
