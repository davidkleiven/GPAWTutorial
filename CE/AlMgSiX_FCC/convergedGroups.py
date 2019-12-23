from ase.db import connect

DB_NAME = 'almgsiX_dft.db'
FNAME = 'converged.txt'
def convergedGroups():
    db = connect(DB_NAME)
    groups = set()
    for row in db.select():
        g = row.get('group', -1)
        if g != -1:
            groups.add(g)

    converged = []
    for g in groups:
        num = sum(1 for _ in db.select(group=g))
        if num == 3:
            converged.append(g)
    print(converged)
    
    with open(FNAME, 'w') as of:
        for c in converged:
            of.write('{}\n'.format(c))
    print("Converged groups written to {}".format(FNAME))

    

convergedGroups()
