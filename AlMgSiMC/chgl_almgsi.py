#from cemc.phasefield import PyCHGL
from phasefield_cxx import PyCHGL
from phasefield_cxx import PyCHGLRealSpace
from phasefield_cxx import PyKernelRegressor
from phasefield_cxx import PyGaussianKernel
from phasefield_cxx import PyPolynomial
from phasefield_cxx import PyTwoPhaseLandau
from phasefield_cxx import PyPolynomialTerm
from phasefield_cxx import PyKhachaturyan
from matplotlib import pyplot as plt
from cemc.tools import to_full_rank4
import numpy as np
from cemc.phasefield.tools import get_polyterms
import json

FNAME = "chgl_almgsi.json"

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
    
    C_al = to_full_rank4(C_al)
    # Misfit strain in 3D
    misfit = np.array([[ 0.0440222,   0.00029263,  0.0008603 ],
                        [ 0.00029263, -0.0281846,   0.00029263],
                        [ 0.0008603,   0.00029263,  0.0440222 ]])

    misfit1 = misfit = np.array([[ 0.0440222,   0.00029263,  0.0 ],
                        [ 0.00029263, -0.0281846,   0.0],
                        [ 0.0,   0.0,  0.0440222 ]])

    khac1 = PyKhachaturyan(2, C_al, misfit1)

    misfit2 = np.array([[ -0.0281846,   0.00029263,  0.0 ],
                        [ 0.00029263, 0.0440222,   0.0],
                        [ 0.0,   0.0,  0.0440222 ]])
    khac2 = PyKhachaturyan(2, C_al, misfit2)

    chgl.add_strain_model(khac1, 1)
    chgl.add_strain_model(khac2, 2)


def check_loaded_polynomial(poly, regressor):
    # As function of concentration
    c = np.linspace(0.0, 1.0, 10000)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1)

    for n_eq in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        energy = [poly.evaluate(c[i], [n_eq, 0.0]) for i in range(len(c))]
        conc_deriv = [poly.partial_deriv_conc(c[i], [n_eq, 0.0]) for i in range(len(c))]
        ax3.plot(c, conc_deriv)
        ax.plot(c, energy)

    # Consider derivative with respect to n_eq
    n_eq = np.linspace(0.0, 1.0, 1000)
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(1, 1, 1)
    for c in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        shape_deriv = [poly.partial_deriv_shape(c, [n_eq[i], 0.0], 0) for i in range(len(n_eq))]
        ax4.plot(n_eq, shape_deriv)

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(1, 1, 1)
    # ax2.plot(c, [regressor.evaluate(c[i]) for i in range(len(c))])
    ax3.axhline(0.0)
    plt.show()


def main():
    #prefix = "data/almgsi_chgl_random_seed_strain_noise2/chgl"
    prefix = "data/almgsi_chgl_strain_large_cell/chgl"
    dim = 2
    L = 1024
    num_gl_fields = 2
    M = 0.1
    alpha = 5.0
    dt = 1E-4
    gl_damping = M

    coeff, terms = get_polyterms(FNAME)

    poly = PyPolynomial(3)

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
        if powers[-1] == 0:
            poly.add_term(c, PyPolynomialTerm(powers[:-1]))
            print(c, powers)

    alpha = grad_coeff[0]
    #alpha = 5.0
    gradient_coeff = [[grad_coeff[1], grad_coeff[2]],
                      [grad_coeff[2], grad_coeff[1]]]

    print(gradient_coeff)

    chgl = PyCHGL(dim, L, prefix, num_gl_fields, M, alpha, dt,
                  gl_damping, gradient_coeff)

    chgl = PyCHGLRealSpace(dim, L, prefix, num_gl_fields, M, alpha, dt,
                  gl_damping, gradient_coeff)
    
    landau = PyTwoPhaseLandau()
    landau.set_polynomial(poly)
    landau.set_kernel_regressor(regressor)
    landau.set_discontinuity(info["discontinuity_conc"], info["discontinuity_jump"])

    check_loaded_polynomial(landau, regressor)
    # exit()
    chgl.set_free_energy(landau)
    # chgl.use_HeLiuTang_stabilizer(1.0)
    # chgl.random_initialization([0.0, 0.0, 0.0], [1.0, 0.85, 0.85])
    # chgl.from_npy_array(droplet_at_center(L))
    # chgl.from_npy_array(precipitates(L))
    #chgl.from_npy_array(random_consistent(L, mean_conc=0.01))
    chgl.from_npy_array(precipitates_circles(L, 50))
    chgl.use_adaptive_stepping(1E-10, 4, 0.01)
    # chgl.set_filter(1.0)
    chgl.build2D()
    add_strain(chgl)
    #chgl.set_cook_noise(1E-2)
    chgl.from_file(prefix + "00000100000.grid")

    #chgl.save_noise_realization(prefix + "noise.grid", 0)
    #exit()

    chgl.run(50000, 1000, start=100000)
    chgl.save_free_energy_map(prefix+"_free_energy_map.grid")

def droplet_at_center(L):
    from scipy.ndimage import gaussian_filter
    from matplotlib import pyplot as plt
    c = np.zeros((L, L))
    dr_start = int(3*L/8)
    dr_end = int(5*L/8)
    c[dr_start:dr_end, dr_start:dr_end] = 0.9
    n_eq = np.zeros((L, L))
    n_eq[dr_start:dr_end, dr_start:dr_end] = 0.8
    n_eq2 = np.zeros((L, L))

    c = gaussian_filter(c, 2)
    n_eq = gaussian_filter(n_eq, 2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(c)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(n_eq)
    plt.show()
    #exit()

    return [c, n_eq, n_eq2]

def precipitates(L):
    from matplotlib import pyplot as plt
    size = L/5
    num = 3
    c = np.zeros((L, L))
    n_eq1 = np.zeros((L, L))
    n_eq2 = np.zeros((L, L))
    for i in range(num):
        x = np.random.randint(size, L-size)
        y = np.random.randint(size, L-size)

        x0 = x - int(size/2)
        x1 = x + int(size/2)
        y0 = y - int(size/2)
        y1 = y + int(size/2)
        c[x0:x1, y0:y1] = 0.9

        if (np.random.rand() < 0.5):
            n_eq1[x0:x1, y0:y1] = 0.9
        else:
            n_eq2[x0:x1, y0:y1] = 0.9
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(c)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(n_eq1)
    plt.show()
    return [c, n_eq1, n_eq2]


def random_consistent(L, mean_conc=0.5):
    from itertools import product
    if mean_conc > 0.5:
        scale = 1.0 - mean_conc
    else:
        scale = mean_conc
    c = 2*scale*np.random.rand(L, L)

    if (mean_conc > 0.5):
        c = 1.0 - c
    n_eq1 = np.zeros((L, L))
    n_eq2 = np.zeros((L, L))
    #return [c, n_eq1, n_eq2]
    return insert_seed([c, n_eq1, n_eq2])


def insert_seed(arrays):
    L = arrays[0].shape[0]
    size = L/5.0

    arrays[0][50:150, 50:150] = 1.0
    arrays[1][50:150, 50:150] = 0.8
    return arrays

def precipitates_circles(L, R):
    from matplotlib import pyplot as plt
    size = R
    num = 10
    num_inserted = 0
    max_attempts = 1000
    c = np.zeros((L, L))
    n_eq1 = np.zeros((L, L))
    n_eq2 = np.zeros((L, L))
    for i in range(max_attempts):
        x = np.random.randint(size, L-size)
        y = np.random.randint(size, L-size)

        eta = n_eq1
        if (np.random.rand() < 0.5):
            eta = n_eq2
        
        bb = c[x-R:x+R, y-R:y+R]
        eta_loc = eta[x-R:x+R, y-R:y+R]
        if np.any(bb) > 1E-6:
            continue
        
        x_loc, y_loc = np.meshgrid(range(bb.shape[0]), range(bb.shape[1]))
        x0 = bb.shape[0]/2
        y0 = bb.shape[1]/2
        r = np.sqrt((x_loc-x0)**2 + (y_loc-y0)**2)
        bb[r <= R] = 1.0
        eta_loc[r <= R] = 0.8
        num_inserted += 1

        if num_inserted >= num:
            break
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(c)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(n_eq1)
    plt.show()
    #exit()
    return [c, n_eq1, n_eq2]

if __name__ == "__main__":
    main()
