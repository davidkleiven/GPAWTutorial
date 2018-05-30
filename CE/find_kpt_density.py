from gpaw import GPAW,PW
from ase.build import bulk

atoms = bulk("Al",a=4.05)

kpts = {"density":1.37,"even":True}
calc = GPAW(mode=PW(500), xc="PBE", kpts=kpts, nbands=-10)
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
