from clease.settings import settingsFromJSON
from clease.calculator import attach_calculator
from clease.montecarlo.observers import Snapshot, MCObserver, EntropyProductionRate
from clease.montecarlo import KineticMonteCarlo, SSTEBarrier, NeighbourSwap
import json
import random
from ase.io.trajectory import TrajectoryWriter, TrajectoryReader
from clease.montecarlo import Montecarlo
from ase.neighborlist import neighbor_list
import matplotlib as mpl
mpl.rcParams.update({'font.family': 'serif', 'font.size': 11})
from matplotlib import pyplot as plt
import numpy as np


class SnapshotNoAl(MCObserver):
    def __init__(self, fname, atoms):
        self.tw = TrajectoryWriter(fname)
        self.atoms = atoms

    def __call__(self, change):
        at = self.atoms.copy()
        for i in range(len(at), 0, -1):
            if at[i-1].symbol == 'Al':
                del at[i-1]
        self.tw.write(at)

    def close(self):
        self.tw.close()


def main():
    kmc_run = True
    eci = {}
    with open("data/almgsix_normal_ce.json", 'r') as infile:
        data = json.load(infile)
        eci = data['eci']

    settings = settingsFromJSON("data/settings_almgsiX_voldev.json")
    settings.basis_func_type = "binary_linear"
    initial = settings.atoms*(10, 10, 10)
    atoms = attach_calculator(settings, initial.copy(), eci)

    # Insert Mg and Si
    for s in ['Mg', 'Si']:
        for _ in range(40):
            idx = random.randint(0, len(atoms)-1)
            atoms[idx].symbol = s
    vac_idx = 80
    atoms[vac_idx].symbol = 'X'

    T = 100
    barrier = SSTEBarrier(
        {
            'Al': 0.6,
            'Mg': 0.6,
            'Si': 0.6
        }
    )

    snapshot = SnapshotNoAl("data/kmc_test100.traj", atoms)
    neighbor = NeighbourSwap(atoms, 3.0)
    for l in neighbor.nl:
        assert len(l) == 12

    if kmc_run:
        kmc = KineticMonteCarlo(
            atoms, T, barrier, [neighbor]
        )
        kmc.epr = EntropyProductionRate(buffer_length=1000)
        kmc.attach(snapshot, 100)
        kmc.run(10000, vac_idx)
    else:
        mc = Montecarlo(atoms, T)
        mc.attach(snapshot, 100)
        mc.run(steps=100000)
    snapshot.close()

def cluster_size(fname):
    traj = TrajectoryReader(fname)
    sizes = []
    num_neigh = 6
    for atoms in traj:
        first, second = neighbor_list('ij', atoms, cutoff=3.0)
        cluster_size = 0
        n_count = [0 for _ in range(len(first))]
        for f, s in zip(first, second):
            if atoms[f].symbol == 'Mg' and atoms[s].symbol == 'Si':
                n_count[f] += 1
            elif atoms[f].symbol == 'Si' and atoms[s].symbol == 'Mg':
                n_count[f] += 1
        cluster_size = sum(1 for n in n_count if n >= num_neigh)
        sizes.append(cluster_size)
    
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(sizes)
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction of solutes in cluster")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig("data/kmc_growth.png", dpi=200)
    plt.show()

def entropy_prod():
    data = np.loadtxt("epr.txt")
     
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.cumsum(data))
    ax.set_xlabel("Time")
    ax.set_ylabel("Entropy")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    plt.show()


#main()
#cluster_size("data/kmc_test.traj")
entropy_prod()
