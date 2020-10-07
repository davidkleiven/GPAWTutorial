from clease.settings import CEBulk, Concentration
from clease.tools import reconfigure

db_name = "data/cupd_ce.db"

def main():
    conc = Concentration(basis_elements=[['Cu', 'Pd']])
    settings = CEBulk(conc, crystalstructure='fcc', a=4.05, size=[1, 1, 1], db_name=db_name,
        max_cluster_dia=[5.0, 5.0, 5.0])
    settings.save("data/cupd_settings.json")
    reconfigure(settings)

main()