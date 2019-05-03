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

def main():
    prefix = "data/almgsi_chgl_circles/chgl"
    dim = 2
    L = 1024
    R = 50
    num_gl_fields = 2
    M = 0.1
    alpha = 5.0
    dt = 0.1
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
    chgl.set_free_energy(landau)
    chgl.from_npy_array(precipitates(L, R))
    chgl.use_adaptive_stepping(1E-10, 4, 0.01)
    # chgl.set_filter(1.0)
    chgl.build2D()

    chgl.run(100000, 1000, start=0)
    chgl.save_free_energy_map(prefix+"_free_energy_map.grid")

    return [c, n_eq, n_eq2]

def precipitates(L, R):
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
    size = L/20.0

    arrays[0][50:70, 50:100] = 1.0
    arrays[1][50:70, 50:100] = 0.8
    return arrays

if __name__ == "__main__":
    main()
