from cemc.mcmc import SoluteChainMC, FixedNucleusMC
from cemc.mcmc import Snapshot, FixEdgeLayers
from ase.clease import CEBulk, Concentration
from cemc import get_atoms_with_ce_calc
from cemc.mcmc.strain_energy_bias import Strain
from inertia_barrier import get_nanoparticle, insert_nano_particle
from cemc.mcmc import MCBackup
import json
import numpy as np
from ase.geometry import get_layers
from cemc.mcmc import ConstrainElementByTag

def atoms_with_calc(N):
    conc = Concentration(basis_elements=[["Al", "Mg", "Si"]])

    kwargs = {
        "crystalstructure": "fcc",
        "a": 4.05,
        "size": [4, 4, 4],
        "concentration": conc,
        "db_name": "data/almgsi_10dec.db",
        "max_cluster_size": 4
    }

    ceBulk = CEBulk(**kwargs)
    eci_file = "data/almgsi_fcc_eci_newconfig_dec10.json"
    with open(eci_file, 'r') as infile:
        ecis = json.load(infile)
    db_name = "large_cell_db{}x{}x{}_dec10.db".format(N, N, N)
    #ecis = {"c1_0": 1.0}
    atoms = get_atoms_with_ce_calc(ceBulk, kwargs, ecis, size=[N, N, N], db_name=db_name)

    symbs = [atom.symbol for atom in atoms]
    diag = atoms.get_cell().T.dot([0.5, 0.5, 0.5])
    pos = atoms.get_positions() - diag
    lengths = np.sum(pos**2, axis=1)
    indx = np.argmin(lengths)
    symbs[indx] = "Mg"
    atoms.get_calculator().set_symbols(symbs)
    return atoms

def tag_by_layer_type(atoms):
    layers, dists = get_layers(atoms, (1, 0, 1))
    for atom in atoms:
        atom.tag = layers[atom.index]%2
    return atoms

def get_strain_observer(mc):
    mu = 0.27155342515650893
    C_al = C_al = np.array([[0.62639459, 0.41086487, 0.41086487, 0, 0, 0],
            [0.41086487, 0.62639459, 0.41086487, 0, 0, 0],
            [0.41086487, 0.41086487, 0.62639459, 0, 0, 0],
            [0, 0, 0, 0.42750351, 0, 0],
            [0, 0, 0, 0, 0.42750351, 0],
            [0, 0, 0, 0, 0, 0.42750351]])
    C_al = np.loadtxt("data/C_al.csv", delimiter=",")
    C_mgsi = np.loadtxt("data/C_MgSi100.csv", delimiter=",")
    misfit = np.array([[ 0.0440222,   0.00029263,  0.0008603 ],
              [ 0.00029263, -0.0281846,   0.00029263],
              [ 0.0008603,   0.00029263,  0.0440222 ]])
    return Strain(mc_sampler=mc, cluster_elements=["Mg", "Si"], C_matrix=C_al, C_prec=C_mgsi,
                  misfit=misfit, poisson=mu)

def main():
    atoms = atoms_with_calc(50)
    # mc = SoluteChainMC(atoms, 350, cluster_elements=["Mg", "Si"], 
    #     cluster_names=["c2_01nn_0"])

    T = [10]
    snapshot = Snapshot(trajfile="data/solute_chain{}_nostrain.traj".format(T[0]), atoms=atoms)
    first = True
    nano_part = get_nanoparticle()
    for temp in T:
        print("Current temperature {}".format(T))
        mc = FixedNucleusMC(
            atoms, temp, network_name=["c2_01nn_0"],
            network_element=["Mg", "Si"],
            max_constraint_attempts=1E6)
        backup = MCBackup(mc, overwrite_db_row=False, db_name="data/mc_solute_no_strain.db")
        mc.attach(backup, interval=20000)
        strain = get_strain_observer(mc)
        mc.add_bias(strain)

        if first:
            first = False
            #mc.grow_cluster({"Mg": 1000, "Si": 1000})
            symbs = insert_nano_particle(atoms.copy(), nano_part)
            mc.set_symbols(symbs)
            tag_by_layer_type(mc.atoms)
            cnst = ConstrainElementByTag(atoms=mc.atoms, element_by_tag=[["Mg", "Al"], ["Si", "Al"]])
            mc.add_constraint(cnst)

        #mc.build_chain({"Mg": 500, "Si": 500})
        fix_layer = FixEdgeLayers(thickness=5.0, atoms=mc.atoms)
        mc.add_constraint(fix_layer)
        mc.attach(snapshot, interval=20000)
        mc.runMC(steps=2000000, init_cluster=False)

if __name__ == "__main__":
    main()

