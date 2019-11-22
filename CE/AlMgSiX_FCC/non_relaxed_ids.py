from ase.db import connect

DB_NAME = 'almgsiX_dft.db'

def non_relaxed():
    groups = set()
    db = connect(DB_NAME)
    for row in db.select():
        groups.add(row.get('group', -1))
    
    ids = []
    for group in groups:
        gpw_15_relaxed = sum(1 for row in db.select(group=group, type='full_relax_gpw15'))

        if gpw_15_relaxed >= 1:
            continue
        
        try:
            selection = [('group', '=', group), ('type', '=', 'volume_relax')]
            num = sum(1 for _ in db.select(selection=selection))
            if num == 0:
                continue
            uid = list(db.select(selection=selection))[0].id
            ids.append(uid)
        except KeyError:
            continue
    print(sorted(ids))


def non_relaxed_volume():
    groups = set()
    groups = set()
    db = connect(DB_NAME)
    for row in db.select():
        groups.add(row.get('group', -1))
    
    ids = []
    for group in groups:
        vol_relaxed = sum(1 for row in db.select(group=group, type='volume_relax'))

        if vol_relaxed == 0:
            ids.append(db.get(group=group, type='initial').id)
    print(ids)

def non_relaxed_volume2():
    groups = set()
    db = connect(DB_NAME)
    ids = []
    for row in db.select(struct_type='initial'):
        group = row.get('group', -1)
        if group == -1:
            ids.append(row.id)
        else:
            vol_relaxed = sum(1 for _ in db.select(group=group, type='volume_relax'))
            if vol_relaxed == 0:
                print(group, row.id)
                ids.append(row.id)
    print(ids)



def demo_transfer():
    trans_db_name = 'demo_transfer.db'
    db = connect(DB_NAME)
    row = list(db.select())[0]
    transfer_db = connect(trans_db_name)
    transfer_db.write(row)
    transfer_db.write(row)

non_relaxed()
#non_relaxed_volume2()
#demo_transfer()
