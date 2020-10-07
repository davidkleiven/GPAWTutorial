from clease.settings import settingsFromJSON
from clease.calculator import attach_calculator
from clease.montecarlo.observers import Snapshot, MCObserver
from clease.montecarlo import KineticMonteCarlo, LocalEnvironmentBarrier, NeighbourSwap
import json
import random
from ase.io.trajectory import TrajectoryWriter

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
            idx = random.randint(0, len(atoms))
            atoms[idx].symbol = s
    vac_idx = 80
    atoms[vac_idx].symbol = 'X'

    T = 300
    barrier = LocalEnvironmentBarrier(
        {
            'Al': 0.6,
            'Mg': 0.6,
            'Si': 0.6
        }
    )

    snapshot = SnapshotNoAl("data/kmc_test.traj", atoms)
    neighbor = NeighbourSwap(atoms, 3.0)
    for l in neighbor.nl:
        assert len(l) == 12

    kmc = KineticMonteCarlo(
        atoms, T, barrier, [neighbor]
    )
    kmc.add_observer(snapshot, 100)
    kmc.run(1000000, vac_idx)
    snapshot.close()

main()
