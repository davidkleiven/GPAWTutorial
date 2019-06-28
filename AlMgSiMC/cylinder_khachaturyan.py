import numpy as np
from apal import Khachaturyan
import matplotlib as mpl
mpl.rcParams.update({'font.size': 18, 'axes.unicode_minus': False, 'svg.fonttype': 'none'})
from matplotlib import pyplot as plt

C_al = np.array([[0.62639459, 0.41086487, 0.41086487, 0, 0, 0],
                 [0.41086487, 0.62639459, 0.41086487, 0, 0, 0],
                 [0.41086487, 0.41086487, 0.62639459, 0, 0, 0],
                 [0, 0, 0, 0.42750351, 0, 0],
                 [0, 0, 0, 0, 0.42750351, 0],
                 [0, 0, 0, 0, 0, 0.42750351]])

SIZE = 512

MISFIT = np.array([[0.0440222, 0.00029263, 0.0008603],
                   [0.00029263, -0.0281846, 0.00029263],
                   [0.0008603, 0.00029263, 0.0440222]])


def strain_energy(radius, length):
    from cylinder import create_cylinder
    khach = Khachaturyan(elastic_tensor=C_al, misfit_strain=MISFIT)

    voxels = np.zeros((SIZE, SIZE, SIZE), dtype=np.int32)
    voxels = create_cylinder(voxels, radius, length, SIZE)
    print("Created cylinder")

    energy = khach.strain_energy_voxels(voxels)
    print("Strain energy: {} meV/A^3".format(energy*1000))
    return energy*1000.0


def strain_ellipsoid(a, b, c):
    from cylinder import create_ellipsoid
    khach = Khachaturyan(elastic_tensor=C_al, misfit_strain=MISFIT)

    voxels = np.zeros((SIZE, SIZE, SIZE), dtype=np.int32)
    voxels = create_ellipsoid(voxels, a, b, c, SIZE)
    print("Created ellipsoid")

    energy = khach.strain_energy_voxels(voxels)
    print("Strain energy: {} meV/A^3 (a={},b={},c={})".format(energy*1000, a, b, c))
    return energy*1000.0


def calculate_all():
    r = 20
    data = []
    for d in range(2, 200, 4):
        energy = strain_energy(r, d)
        data.append([r, d, energy])
    fname = "data/strain_energy_cylinder{}.csv".format(int(r))
    np.savetxt(fname, data, delimiter=",", header="Radius (A), Length (A), Energy (meV/A^3)")


def calculate_ellipsoid():
    a = c = 20
    data = []
    flip_ba = True
    for b in list(range(2, 20, 4)) + list(range(20, 200, 20)):
        if flip_ba:
            energy = strain_ellipsoid(b, a, c)
        else:
            energy = strain_ellipsoid(a, b, c)
        data.append([a, b, c, energy])
    
    if flip_ba:
        fname = "data/strain_energy_ellipsoid{}_flipped.csv".format(int(a))
    else:
        fname = "data/strain_energy_ellipsoid{}.csv".format(int(a))
    np.savetxt(fname, data, delimiter=",", header="Half-axis x (A), Half-axis y (A), Half-axis z (A), Energy (meV/A^3)")


def save_voxels(radius, length):
    from cylinder import create_cylinder
    voxels = np.zeros((SIZE, SIZE, SIZE), dtype=np.int32)
    voxels = create_cylinder(voxels, radius, length, SIZE)
    voxels = np.array(voxels, dtype=np.uint8)
    fname = "/work/sophus/cylinder_R{}_L{}.bin".format(int(radius), int(length))
    voxels.tofile(fname)
    print("Voxels written to {}".format(fname))

def plot_strain_energy(fname):
    data = np.loadtxt(fname, delimiter=",")
    aspect = data[:, 1]/data[:, 0]
    energy = data[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(aspect, energy, color="#5d5c61")
    ax.set_xlabel("Aspect ratio (L/R)")
    ax.set_ylabel(r"Strain energy (meV/\r{A}\$^3\$)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

def plot_strain_energy_ellipsoids():
    data = np.loadtxt("data/strain_energy_ellipsoid20.csv", delimiter=",")
    data_flipped = np.loadtxt("data/strain_energy_ellipsoid20_flipped.csv", delimiter=",")
    aspect = data[:, 1]/data[:, 0]
    aspect_flipped = data_flipped[:, 1]/data_flipped[:, 0]
    energy = data[:, 3]
    energy_flipped = data_flipped[:, 3]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(aspect, energy, color="#5d5c61", marker="o", mfc="none")
    ax.plot(aspect_flipped, energy_flipped, color="#557a95", marker="v", mfc="none")
    ax.set_xlabel("Aspect ratio (L/R)")
    ax.set_ylabel(r"Strain energy (meV/\r{A}\$^3\$)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

#calculate_all()
#calculate_ellipsoid()
plot_strain_energy_ellipsoids()
#plot_strain_energy("data/strain_energy_cylinder20.csv")
#save_voxels(50, 400)
