from ase.ce.settings_bulk import BulkCrystal
from ase.ce.corrFunc import CorrFunction

db_name = "ternay_demo.db"
conc_args = {
    "conc_ratio_min_1":[[1,0,0,0]],
    "conc_ratio_max_1":[[0,1,0,0]],
}
N = 4
ceBulk = BulkCrystal( crystalstructure="fcc", a=4.05, size=[N,N,N], \
                      basis_elements=[["Al","Mg","Si","Cu"]], conc_args=conc_args,
                      db_name=db_name,
                      max_cluster_size=4)
ceBulk._get_cluster_information()
cf = CorrFunction(ceBulk)
cfs = cf.get_cf(ceBulk.atoms)
