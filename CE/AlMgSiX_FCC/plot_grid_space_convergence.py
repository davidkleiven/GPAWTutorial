from ase.db import connect
import numpy as np
from matplotlib import pyplot as plt

def main():
    db = connect("grid_space_conv.db")
    symbs = ["Al", "Mg", "Si"]
    data = {symb: {"h":[], "energy": []} for symb in symbs}
    for row in db.select():
        print(row.formula)
        data[row.formula]["h"].append(row["grid_spacing"])
        data[row.formula]["energy"].append(row["energy"])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for k, v in data.items():
        indx = np.argsort(v["h"])
        h = np.array(v["h"])[indx]
        E = np.array(v["energy"])[indx]
        E -= E[-1]
        ax.plot(h, E, marker="o", label=k)

    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
