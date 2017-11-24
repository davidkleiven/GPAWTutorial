import sys
sys.path.append("/home/davidkl/Documents/MICE/mice")
sys.path.insert(1,"/home/davidkl/Documents/aseJin")
from ase.ce.settings import BulkCrystal
from correlation_function import LagrangeBasisCorrFunc
from lagrange_evaluate import LagrangeEvaluator
from solver import Solver
from ase.visualize import view
import numpy as np

def main():
    N = 4
    comp_sweep = False
    chem_pot_sweep = True
    db_name = "ce_hydrostatic.db"
    conc_args = {
            "conc_ratio_min_1":[[20,44]],
            "conc_ratio_max_1":[[64,0]],
        }
    ceBulk = BulkCrystal( "fcc", 4.05, [N,N,N], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4 )
    evaluator = LagrangeEvaluator( ceBulk, lamb=8E-6, penalty="l1" )
    evaluator.change_basis(ceBulk, verbose=True, io="save", fname="corr_func_hydro.csv")
    evaluator.plot_energy()
    eci = evaluator.get_cluster_name_eci_dict
    print (eci)

    if ( comp_sweep ):
        composition_sweep(ceBulk,eci)
    elif ( chem_pot_sweep ):
        N = 4
        #ceBulk = BulkCrystal( "fcc", 4.05, [N,N,N], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4 )
        #ceBulk.cluster_names, dummy2, ceBulk.cluster_indx = ceBulk.get_cluster_information()
        #ceBulk.trans_matrix = ceBulk.create_translation_matrix()
        chemical_potential_sweep( ceBulk, eci, np.linspace(0.0,1.0,10) )
    else:
        atoms = ceBulk.atoms
        fraction = 0.39
        for i in range(0,int(fraction*len(atoms))):
            atoms[i].symbol = "Mg"
        solver = Solver(ceBulk,eci,verbose=True,chem_pot={"Mg":0.4991415257506623})
        model = solver.get_model()

        solver.solve( "gurobi", model, atoms_file="atoms_hydro.xyz", show_atoms=True )


def chemical_potential_sweep( ceBulk, eci, chem_pots ):
    solver = Solver(ceBulk,eci,verbose=True,chem_pot={"Mg":chem_pots[0]} )
    ceBulk.atoms[0].symbol = "Mg"
    model = solver.get_model()

    lower = 0.49
    upper = 0.5
    # Find upper and lower bounds for pure phase
    for i in range(0,100):
        current = (lower+upper)/2.0
        model = solver.update_chemical_potentials(model,{"Mg":current})

        # Bisection
        solver.solve( "gurobi", model )
        elms = solver.get_atoms_count()
        if ( elms["Mg"] >= 32 ):
            lower = current
        else:
            upper = current
        if ( elms["Mg"] <= 50 and elms["Mg"] >= 10 ):
            break
    print (lower,upper)

    #model = solver.update_chemical_potentials( model, {"Mg":current})
    #solver.solve( "gurobi", model, show_atoms=True )

def composition_sweep(ceBulk,eci):
    atoms_ref = ceBulk.atoms.copy()

    compositions = np.linspace(0.01,0.6,10)
    for comp in compositions:
        print ("Composition: %.2E"%(float(comp)))
        for i in range(0,len(ceBulk.atoms)):
            ceBulk.atoms[i].symbol = "Al"
        Nmg = int(float(comp)*len(ceBulk.atoms))
        if ( Nmg == 0 ):
            continue
        for i in range(0,Nmg):
            ceBulk.atoms[i].symbol = "Mg"
        solver = Solver(ceBulk,eci,verbose=True)
        model = solver.get_model()
        solver.solve( "gurobi", model, atoms_file="composition/atoms_comp_sweep_%d.xyz"%(int(100*float(comp))))
        solver.save( model, "composition/info_comp_sweep_%d.txt"%(int(100*float(comp))) )

if __name__ == "__main__":
    main()
