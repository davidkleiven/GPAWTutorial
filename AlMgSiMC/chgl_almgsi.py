#from cemc.phasefield import PyCHGL
from phasefield_cxx import PyCHGL
from cemc.phasefield.tools import get_polyterms


def main():
    prefix = "data/almgsi_chgl/chgl"
    dim = 2
    L = 1024.0
    num_gl_fields = 2
    M = 0.001
    alpha = 5.0
    dt = 0.01
    gl_damping = M
    gradient_coeff = [[1.2942188134363368, 14.13799371386005],
                      [14.13799371386005, 1.2942188134363368]]

    chgl = PyCHGL(dim, L, prefix, num_gl_fields, M, alpha, dt,
                  gl_damping, gradient_coeff)
    coeff, terms = get_polyterms("data/almgsi_chgl/coeff.csv")
    for c, t in zip(coeff, terms):
        chgl.add_free_energy_term(c, t)
    chgl.random_initialization([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    chgl.run(100000, 5000, start=0)

if __name__ == "__main__":
    main()
