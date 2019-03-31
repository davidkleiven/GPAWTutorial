#from cemc.phasefield import PyCHGL
from phasefield_cxx import PyCHGL
from phasefield_cxx import PyKernelRegressor
from phasefield_cxx import PyGaussianKernel
from phasefield_cxx import PyPolynomial
from phasefield_cxx import PyTwoPhaseLandau
from phasefield_cxx import PyPolynomialTerm
from cemc.phasefield.tools import get_polyterms
import json

FNAME = "chgl_almgsi.json"

def main():
    prefix = "data/almgsi_chgl/chgl"
    dim = 2
    L = 128
    num_gl_fields = 2
    M = 0.001
    alpha = 5.0
    dt = 0.1
    gl_damping = M*2

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

    alpha = grad_coeff[0]
    gradient_coeff = [[grad_coeff[1], grad_coeff[2]],
                      [grad_coeff[2], grad_coeff[1]]]

    chgl = PyCHGL(dim, L, prefix, num_gl_fields, M, alpha, dt,
                  gl_damping, gradient_coeff)
    
    landau = PyTwoPhaseLandau()
    landau.set_polynomial(poly)
    landau.set_kernel_regressor(regressor)
    chgl.set_free_energy(landau)
    chgl.random_initialization([0.9, 0.7, 0.7], [1.0, 0.85, 0.85])
    chgl.run(500, 50, start=0)
    chgl.save_free_energy_map(prefix+"_free_energy_map.grid")

if __name__ == "__main__":
    main()
