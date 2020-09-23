from clease.settings import CEBulk, Concentration

db_name = "data/almgsicu_ce.db"

def main():
    conc = Concentration(basis_elements=[['Al', 'Mg', 'Si', 'Cu']])
    conc.set_conc_ranges([[(0, 1), (0, 1), (0, 0.5), (0, 1)]])
    settings = CEBulk(conc, crystalstructure='fcc', a=4.05, size=[1, 1, 1], db_name=db_name,
        max_cluster_dia=[5.0, 5.0, 4.0])
    settings.save("data/almgsicu_settings.json")

main()