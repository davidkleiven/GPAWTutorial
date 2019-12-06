from clease import Concentration, CEBulk
from clease import NewStructures

def main():
    conc = Concentration(basis_elements=[['Al', 'Mg']], A_lb=[[10, 0]], b_lb=[7])
    settings = CEBulk(conc, crystalstructure='fcc', a=4.05, size=[4, 4, 4],
                      max_cluster_size=4, max_cluster_dia=[4.1, 4.1, 4.1],
                      db_name='almg_ce.db')

    # Generate 30 structures
    new_struct = NewStructures(settings, struct_per_gen=30)
    new_struct.generate_random_structures()

main()