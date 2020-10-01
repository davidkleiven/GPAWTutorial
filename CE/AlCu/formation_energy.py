from matplotlib import pyplot as plt
from ase.db import connect
from collections import Counter

# REF_EMT_ENERGIES = {
#     'Al': -0.005,
#     'Cu': -0.007
# }

REF_EMT_ENERGIES = {
    'Ag': 0.0,
    'Pt': 0.0
}

REF_EMT_ENERGIES = {
    'Cu': -0.007,
    'Pd': 0.0
}


def emt_data():
    db = connect("data/cupd.db")
    concs = []
    form_energies = []
    for row in db.select([('struct_type', '=', 'relaxed')]):
        energy = row.energy
        atoms = row.toatoms()
        counter = Counter(atoms.symbols)

        conc1 = counter.get('Cu', 0.0)/len(atoms)
        conc2 = counter.get('Pd', 0.0)/len(atoms)
        dE = energy/len(atoms) - conc1*REF_EMT_ENERGIES['Cu'] - conc2*REF_EMT_ENERGIES['Pd']
        concs.append(conc1)
        form_energies.append(dE)
    return concs, form_energies

def formation_energy():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    al_conc_emt, dE_emt = emt_data()

    ax.plot(al_conc_emt, dE_emt, 'o', mfc='none')
    plt.show()

formation_energy()