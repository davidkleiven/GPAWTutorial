from ase.db import connect
import sys

DB_NAME = "almgsiX_fcc.db"

def id2final():
    db = connect(DB_NAME)
    names = []
    scond = [("calculator", "=", "gpaw"), ("id", ">=", 3040)]
    for row in db.select(scond):
        names.append(row.name)
    names = list(set(names))

    ids = []
    for name in names:
        uid = 0
        for row in db.select(name=name):
            calc = row.get("calculator", "none")
            if calc == "gpaw":
                uid = row.id
            name = row.name
            types = []
            str_types = row.get("struct_type", "intermediate")
            types.append(str_types)

        if "final" not in types:
            ids.append(uid)
    print(ids)

def id2relax():
    db = connect(DB_NAME)
    scond = [("struct_type", "=", "initial"), ("id", ">=", 3000)]
    names = []
    for row in db.select(scond):
        names.append(row.name)
    names = list(set(names))
    ids = []
    print(names)
    for name in names:
        uid = 0
        num = sum(1 for row in db.select(name=name))
        if num == 1:
            uid = db.get(name=name).id
            ids.append(uid)
    print(ids)

if __name__ == "__main__":
    opt = sys.argv[1]
    
    if opt == "final":
        id2final()
    elif opt == "relax":
        id2relax()