from ase.io import read
from atomtools.ase import AutocorrelationFunction
from matplotlib import pyplot as plt


def main():
    atoms = read("data/atoms39_118116.xyz")
    auto_corr = AutocorrelationFunction(atoms=atoms, plane_normal=[1, 0, 0],
                                        symb_dict={"Si": 1, "Mg": 1})
    auto_corr.projected_image(npix=256, cmap="nipy_spectral")
    plt.show()


if __name__ == "__main__":
    main()
