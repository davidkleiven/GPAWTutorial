from ase.db import connect
from ase.build import bulk

DB_NAME = 'kanzaki.db'

atoms = bulk('Al', cubic=True)*(3, 3, 3)
db = connect(DB_NAME)
# db.write(atoms, group=0, comment="Pure aluminium reference")
# atoms[0].symbol = 'Mg'
# db.write(atoms, group=1, comment="Single Mg")

# atoms[0].symbol = 'Si'
# db.write(atoms, group=2, comment="Single Si")