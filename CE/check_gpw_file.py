import sys
from gpaw import restart
import glob

def file_ok(gpw_file):
    try:
        atoms, calc = restart(gpw_file)
        atoms.get_potential_energy()
    except KeyError as exc:
        print(str(exc))
        return False
    return True

def main(folder):
    num_invalid = 0
    for fname in glob.glob(folder+"/*.gpw"):
        if not file_ok(fname):
            print(fname)

if __name__ == "__main__":
    main(sys.argv[1])