import glob
import json
import gpaw as gp

def extract(fname):
    atoms, calc = gp.restart(fname){}
    result["cell"] = atoms.get_cell().tolist()
    result["cellpar"] = atoms.get_cell_lengths_and_angles().tolist()
    return result

def main():
    all_res = {}
    outfname = "almgsi_cellpar.json"
    for fname in glob.glob("/global/work/davidkl/AlMgSi/*.gpw"):
        try:
            res = extract(fname)
            all_res[fname] = res
        except Exception as exc:
            print (str(exc))

    with open( outfname, 'w' ) as outfile:
        json.dump( all_res, outfile, indent=2, separators=(",",":") )
    print ("Data written to {}".format(outfname))

if __name__ == "__main__":
    main()
