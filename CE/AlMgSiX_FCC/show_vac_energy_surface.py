"""
See singleVacancy.py for information on the databases
"""
import dataset
from mayavi import mlab
import numpy as np

DB_NAME = 'sqlite:///data/vacancy_sweep_no_periodic.db'
def main(runID):
    db = dataset.connect(DB_NAME)
    ref_energy = db['ref_energies'].find_one(runID=runID)['energy']

    positions = []
    energies = []
    for item in db['vac_energy'].find(runID=runID):
        pos = [item['X'], item['Y'], item['Z']]
        positions.append(pos)
        energies.append(item['energy'] - ref_energy)

    cluster_pos = []
    symbs = []
    for item in db['sol_pos'].find(runID=runID):
        cluster_pos.append([item['X'], item['Y'], item['Z']])
        value = 1 if item['symbol'] == 'Mg' else 0
        symbs.append(value)
    cluster_pos = np.array(cluster_pos)
    print(cluster_pos)

    positions = np.array(positions)
    print(positions)
    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    mlab.points3d(cluster_pos[:, 0], cluster_pos[:, 1], cluster_pos[:, 2], symbs, scale_mode='none', scale_factor=2.0)
    mlab.points3d(positions[:, 0], positions[:, 1], positions[:, 2], energies, scale_mode='none', scale_factor=2.0, colormap='copper')
    #mlab.pipeline.volume(mlab.pipeline.gaussian_splatter(pts))
    mlab.colorbar()
    # mesh = mlab.pipeline.delaunay3d(pts)
    # print(mesh)
    # surf = mlab.pipeline.surface(mesh)

    #mlab.view(47, 57, 8.2, (0.1, 0.15, 0.14))
    mlab.show()

    


    


main('0xd0649283')