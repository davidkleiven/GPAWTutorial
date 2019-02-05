from matplotlib import pyplot as plt
import numpy as np

def extract(data, ang1=None, ang2=None, tol=1E-3):
    
    if ang1 is None:
        angle = data[:, 1][np.abs(data[:, 0] - ang2) < tol]
        energy = data[:, 2][np.abs(data[:, 0] - ang2) < tol]
    else:
        angle = data[:, 0][np.abs(data[:, 1] - ang1) < tol]
        energy = data[:, 2][np.abs(data[:, 1] - ang1) < tol]
    return angle, energy
    #return angle*180/np.pi, energy

def main():
    import matplotlib as mpl
    mpl.rcParams.update({"svg.fonttype": "none", "font.size": 18, "axes.unicode_minus": False})
    from matplotlib import pyplot as plt
    needle = np.loadtxt("data/orientation_needle.csv", delimiter=",")
    needle = np.loadtxt("data/strain_energy_needle.csv", delimiter=",")
    needle = np.loadtxt("data/strain_energy_interstitialneedle.csv", delimiter=",")
    needle = np.loadtxt("data/mgsi100_orientation_needle_iso.csv", delimiter=",")
    angles, energy = extract(needle, ang2=0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(angles, energy*1000.0, label="\$\\theta = \\SI{0}{\\degree}\$", color="#5D5C61")
    #angles, energy = extract(needle, ang2=np.pi/4.0)
    angles, energy = extract(needle, ang2=90, tol=2)
    ax.plot(angles, energy*1000.0, label="\$\\theta = \\SI{45}{\\degree}\$", color="#557a95")

    plate = np.loadtxt("data/orientation_plate.csv", delimiter=",")
    plate = np.loadtxt("data/strain_energy_plate.csv", delimiter=",")
    plate = np.loadtxt("data/strain_energy_interstitialplate.csv", delimiter=",")
    plate = np.loadtxt("data/mgsi100_orientation_plate_iso.csv", delimiter=",")
    angles, energy = extract(plate, ang2=0)
    ax.plot(angles, energy*1000.0, ls="--", label="\$\\theta = \\SI{0}{\\degree}\$", color="#5D5C61")
    #angles, energy = extract(plate, ang2=np.pi/4.0)
    angles, energy = extract(plate, ang2=45, tol=2)
    ax.plot(angles, energy*1000.0, ls="--", label="\$\\theta = \\SI{45}{\\degree}\$", color="#557a95")
    ax.set_xlabel("Azimuthal angle (deg)")
    ax.set_ylabel("Strain energy (\$\\SI{}{\\milli\\electronvolt\per\\angstrom\\cubed})\$)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    sphere = np.loadtxt("data/orientation_sphere.csv", delimiter=",")
    angles, energy = extract(sphere, ang2=0)
    angles *= 180.0/np.pi
    ax.plot(angles, energy*1000.0, ls="-.", color="#379683", label="Sphere")
    ax.legend(frameon=False)
    #fig.savefig("data/strain_energy_khchaturyan.svg")
    plt.show()

if __name__ == "__main__":
    main()

