from cemc.tools import CanonicalFreeEnergy
import numpy as np
import dataset

db_name = "data/sa_aucu_only_pairs.db"

def update_entries():
    db = dataset.connect("sqlite:///{}".format(db_name))

    # Find all unique compositions in the db
    statement = "SELECT au_conc FROM results"
    concs = []
    for conc in db.query(statement):
        concs.append(float(conc["au_conc"]))
    concs = np.array(concs)
    concs *= 100
    concs = concs.astype(np.int32)
    concs = np.unique(concs)/100.0

    tbl = db["results"]
    # concs = [0.1, 0.3, 0.95]
    for c in list(concs):
        sql = "SELECT id, temperature, energy FROM results WHERE au_conc > {} AND au_conc < {}".format(c-0.01, c+0.01)
        T = []
        U = []
        for res in db.query(sql):
            T.append(res["temperature"])
            U.append(res["energy"]/1000.0)

        comp = {"Au": c, "Cu": 1-c}
        free_eng = CanonicalFreeEnergy(comp)
        T, U, F = free_eng.get(T, U)

        for temp, u, f in zip(list(T), list(U), list(F)):
            # Update the database
            sql = "SELECT id FROM results WHERE au_conc > {} AND au_conc < {} AND temperature={}".format(c-0.01, c+0.01, temp)
            res = db.query(sql)
            uid = res.next()["id"]
            new_cols = {"id": uid, "free_energy": f, "entropy": (u-f)/temp}
            tbl.update(new_cols, ["id"])

if __name__ == "__main__":
    update_entries()
