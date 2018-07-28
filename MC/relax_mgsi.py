import gpaw as gp
from ase.build import bulk
from ase.visualize import view
from ase.optimize.precon import PreconLBFGS
from ase.io import write
from ase.calculators.emt import EMT

atoms = bulk("Mg", crystalstructure="fcc", a=4.1)
atoms = atoms*(2, 2, 2)
si_indx = [1, 2, 4, 7]
for indx in si_indx:
    atoms[indx].symbol = "Si"

calc = gp.GPAW(mode=gp.PW(600), xc="PBE", kpts=(2, 2, 2), nbands=-100)
atoms.set_calculator(calc)

opt = PreconLBFGS(atoms, variable_cell=True)
opt.run(fmax=0.025, smax=0.003)
write("data/relaxed_mgsi.xyz", atoms)
