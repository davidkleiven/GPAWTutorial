from ase.ce.corrFunc import CorrFunction
from ase.ce.settings_bulk import BulkCrystal
from ase.visualize import view
from ase.db import connect
import numpy as np
from ase import Atoms
from map_to_ideal import map_to_ideal

db_name = "ce_hydrostatic.db"
conc_args = {
    "conc_ratio_min_1":[[1,0]],
    "conc_ratio_max_1":[[0,1]],
}

ceBulk = BulkCrystal( crystalstructure="fcc", a=4.05, size=[4,4,4], \
                      basis_elements=[["Al","Mg"]], conc_args=conc_args,
                      db_name=db_name,
                      max_cluster_size=4)

ceBulk._get_cluster_information()
#ceBulk.view_clusters()
view(ceBulk.atoms)
print (ceBulk.cluster_names[0][2])
print (ceBulk.cluster_indx[0][2][3])
nn_clust = ceBulk.cluster_indx[0][2][3]
db = connect( db_name )
atoms = db.get_atoms(id=1328)
atoms = map_to_ideal( db_name, atoms )
for clust in nn_clust:
    tindx = [ceBulk.trans_matrix[clust[0],nn_clust[i][0]] for i in range(len(nn_clust))]
    """
    symbs = [atoms[indx].symbol for indx in tindx]
    pos = np.zeros((12,3))
    all_pos = atoms.get_positions()
    for i,indx in enumerate(tindx):
        pos[i,:] = all_pos[indx,:]
    pos -= pos[0,:]
    cell = atoms.get_cell()
    cluster = Atoms( symbs, positions=pos, cell=cell, pbc=[1,1,1] )
    diag = cell[0,:] + cell[1,:] + cell[2,:]
    cluster.translate(diag/2.0)
    cluster.wrap()
    view(cluster)
    """
    if ( 0 not in tindx ):
        print ("Translation matrix seems to be wrong!" )
#ceBulk.view_clusters()
#ceBulk._get_cluster_information()
view(atoms)
cf = CorrFunction(ceBulk)
cfs = cf.get_cf_by_cluster_names(atoms,["c2_1000_1_00"])
print (cfs)
