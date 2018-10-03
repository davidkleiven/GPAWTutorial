import sys
import dill
from cemc.mcmc import Montecarlo
from matplotlib import pyplot as plt
from ase.io import write

def plot_evolution(fname):
    mc = Montecarlo.load(fname)
    atoms_fname = fname.rpartition(".")[0]+"_final_atoms.xyz"
    write(atoms_fname, mc.atoms)
    energy_obs = None
    for obs in mc.observers:
        if obs[1].name == "EnergyEvolution":
            energy_obs = obs[1]

    if energy_obs is None:
        raise RuntimeError("Did not find an energy evolution observer!")

    energy = energy_obs.energies
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(energy)
    plt.show()


if __name__ == "__main__":
    plot_evolution(sys.argv[1])
