from ase.db import connect

db_name = "pre_beta_simple_cubic_cpy.db"

def verify_that_read_is_possible():
    print("Reading converged entries")
    db = connect(db_name)
    for row in db.select(converged=1):
        print("ID: {} Energy: {}".format(row.id, row.energy))
    print("Reading works!")

def main():
    verify_that_read_is_possible()

    db = connect(db_name)
    ids = []
    for row in db.select(converged=1):
        ids.append(row.id)

    with db:
        for uid in ids:
            print("Updating: {}".format(uid))
            db.update(uid, struct_type='initial')

if __name__ == "__main__":
    main()
