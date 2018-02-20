#!/usr/bin/env python
import sys
from atomtools.ce import ce_phonon_dos as cpd

def main( argv ):
    for arg in argv:
        if ( arg.find("--phonondb=") != -1 ):
            phdb = arg.split("--phonondb=")[-1]
        elif ( arg.find("--asedb=") != -1 ):
            asedb = arg.split("--asedb=")[-1]

    manager = cpd.PhononDOS_DB( phdb )
    manager.sync_with_ase_db( asedb )

if __name__ == "__main__":
    main( sys.argv[1:] )
