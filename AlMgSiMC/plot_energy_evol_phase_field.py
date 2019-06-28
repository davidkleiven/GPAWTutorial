from matplotlib import pyplot as plt
import numpy as np

def main():
    fname = "data/almgsi_chgl_3D_surface_1nm_64_strain_consistent/chgl_trackvalues.csv"
    data = np.loadtxt(fname, delimiter=",")

    time = data[:, 0]/1000
    strain = data[:, 3]
    surf_energy = data[:, 4]
    vol_energy = data[:, 5]

    vol_energy = (vol_energy - vol_energy[0])/vol_energy[0]
    strain = (strain - strain[0])/strain[0]
    surf_energy = (surf_energy - surf_energy[0])/surf_energy[0]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time, vol_energy, label="Volume")
    ax.plot(time, surf_energy, label="Surf")
    ax.plot(time, strain, label="Strain")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()

