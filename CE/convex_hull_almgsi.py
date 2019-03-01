from ase.clease import ConvexHull
from ase.db import connect

def main():
    from matplotlib import pyplot as plt
    db_name = "almgsi.db"

    qhull = ConvexHull(db_name, atoms_per_fu=4)
    qhull.plot()

    #qhull.show_structures_on_convex_hull()
    plt.show()

    db = connect(db_name)

    for row in db.select([("converged", "=", 1), ("calculator", "!=", "gpaw")]):
        en = row.energy/row.natoms

        count = row.count_atoms()

        for k in count.keys():
            count[k] /= float(row.natoms)
        print(qhull.cosine_similarity_convex_hull(count, en), count)

main()