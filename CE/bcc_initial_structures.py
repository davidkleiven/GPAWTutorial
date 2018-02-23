from ase.build import bulk
from ase.io import write
from itertools import combinations_with_replacement, product

def main():
    atoms = bulk("Al","bcc",a=3.3)
    symbs = ["Al","Mg"]
    combs = list( combinations_with_replacement( symbs, 8 ) )
    file_no = 0
    for c in combs:
        atoms_wrk = atoms.copy()
        atoms_wrk = atoms*(2,2,2)
        for i in range(len(c)):
            atoms_wrk[i].symbol = c[i]
        atoms_wrk = atoms_wrk*(2,2,2)
        write( "data/bcc_trial{}.xyz".format(file_no), atoms_wrk )
        file_no += 1

if __name__ == "__main__":
    main()
