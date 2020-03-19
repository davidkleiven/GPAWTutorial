from ase import Atoms
from ase.visualize import view

pairs = Atoms(['Al', 'Al'], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.86]])
triplet = Atoms(['Al', 'Al', 'Al'], positions=[[0.0, 0.0, 0.0], [0.0, 4.05, 0.0], [0.0, 2.025, 2.025]])
quad = Atoms(['Al', 'Al', 'Al', 'Al'], positions=[[0.0, 0.0, 0.0], [0.0, 4.05, 0.0], [0.0, 2.025, 2.025], [2.025, 2.025, 0.0]])
view([pairs, triplet, quad])