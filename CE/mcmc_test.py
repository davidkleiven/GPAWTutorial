import sys
sys.path.insert(1,"/home/davidkl/Documents/aseJin")
from mcmc import montecarlo as mc
from ase.ce.settings import BulkCrystal
from ase.calculators.cluster_expansion.cluster_expansion import ClusterExpansion
from ase.montecarlo.metropolis import Metropolis
from wanglandau.ce_calculator import CE

# Hard coded ECIs obtained from the ce_hydrostatic.db runs
ecis = {'c3_1225_4_1': -0.00028826723864655595,
        'c2_1000_1_1': -0.012304759727020153,
        'c4_1225_7_1': 0.00018000893943061064,
        'c2_707_1_1': 0.01078731693580544,
        'c4_1225_3_1': 0.00085623111812932343,
        'c2_1225_1_1': -0.010814400169849577,
        'c1_1': -1.0666948263880078,
        'c4_1000_1_1': 0.0016577886586285448,
        'c4_1225_2_1': 0.01124654696678576,
        'c3_1225_2_1': -0.017523737495758165,
        'c4_1225_6_1': 0.0038879587131474451,
        'c4_1225_5_1': 0.00060830459771275532,
        'c3_1225_3_1': -0.011318935831421125,
        u'c0': -2.6466290360293874}

crystalstructure = 'fcc'
alat = 4.05
clat = None
cell_dim = [7,7,7]
num_sites = 1
site_elements = [['Al', 'Mg']]
conc_args = {}
conc_args['conc_ratio_min_1'] = [[1, 0]]
conc_args['conc_ratio_max_1'] = [[0, 1]]
db_name = 'ce_hydrostatic.db'
max_cluster_size = 4
max_cluster_dia = 1.5*alat
#max_cluster_dia = None
reconf_db = False
temperature = 10
num_steps = 10

fcc = BulkCrystal(crystalstructure, alat, clat, cell_dim,
                            num_sites, site_elements, conc_args, db_name,
                            max_cluster_size, max_cluster_dia, reconf_db)
print (fcc.cluster_names)
init_cf = {key:1.0 for key in ecis.keys()}

calc = CE( fcc, ecis, initial_cf=init_cf )
fcc.atoms.set_calculator( calc )

for i in range(20):
    fcc.atoms._calc.update_cf( (i,"Al","Mg") )

def python_mcmc():
    calc = ClusterExpansion(fcc, ecis, None, logfile='CE.log')

    # Assuming that you have atoms object defined here.
    # You can get one from your database
    fcc.atoms.set_calculator(calc)
    print (calc.CF.get_cf_by_cluster_names(fcc.atoms,ecis.keys()))

    MMC = Metropolis( fcc.atoms, temp=temperature, constraint=None, logfile=None)
    energy_sum, eng_squared = MMC.run( num_steps=num_steps, average=True )
    print (energy_sum)

def cpp_mcmc():
    print (calc.updater.get_cf())
    mc_obj = mc.Montecarlo( fcc.atoms, temperature )
    mc_obj.runMC( steps=num_steps )
    thermo = mc_obj.get_thermodynamic()
    print (thermo["energy"])

#python_mcmc()
cpp_mcmc()
