import sqlite3
from sklearn.mixture import GaussianMixture
import numpy as np


db_name = "data/almgsi_mc_sgc.db"
def get_data():
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    sql = "SELECT Al_conc,Mg_conc,Si_conc,mu_c1_0,mu_c1_1,mu_c1_2 "
    sql += "FROM random_direction_sa WHERE temperature=200"
    cur.execute(sql)
    keys = ['Al', 'Mg', 'Si', 'mu0', 'mu1', 'mu2']
    res = {k: [] for k in keys}
    for item in cur.fetchall():
        for i, k in enumerate(keys):
            res[k].append(item[i])
    con.close()
    return res

def classify():
    data = get_data()
    X = np.array([data['Al'], data['Mg'], data['Si']]).T
    
    best_gmm = None
    best_bic = np.infty
    best_num_comp = 0
    for n_comp in range(5, 15):
        gmm = GaussianMixture(n_components=n_comp)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_num_comp = n_comp
    print(f"Number of phases: {best_num_comp}")
    y = best_gmm.predict(X)
    fname = "data/phase_classification.csv"
    with open(fname, 'w') as out:
        out.write("x coord, y coord, z coord, scalar\n")
        for i in range(len(y)):
            out.write(
                f"{data['mu0'][i]}, {data['mu1'][i]}, {data['mu2'][i]}, {y[i]}\n")
    print(f"Data written to {fname}")

    for i in list(np.unique(y)):
        mask = y == i
        avg = np.mean(X[mask, :], axis=0)
        print(f"Class {i} conc {avg}")

if __name__ == '__main__':
    classify()