from ase.build import bulk
from ase.spacegroup import crystal
from wulffpack import SingleCrystal
from ase.visualize import view
from ase.spacegroup import get_spacegroup
from ase.build import cut

gamma1 = 0.33
gamma2 = 0.57
surface_energies = {(0, 0, 1): gamma1, (1, 0, 0): gamma2}
prim = bulk('Al', crystalstructure='fcc', cubic=True)
prim.symbols[[0, 3]] = 'Mg'
prim.symbols[[2, 1]] = 'Si'
sg = get_spacegroup(prim)
prim_cell = sg.scaled_primitive_cell
prim = cut(prim, a=prim_cell[0], b=prim_cell[1], c=prim_cell[2])
from ase.visualize import view
view(prim)

particle = SingleCrystal(surface_energies, primitive_structure=prim, natoms=1000)
view(particle.atoms)
