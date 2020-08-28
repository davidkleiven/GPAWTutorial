from ase.build import bulk
from clease.tools import wrap_and_sort_by_position
from ase.geometry import get_layers
import numpy as np
from ase.visualize import view
import dataset
from clease import settingsFromJSON
from clease.calculator import attach_calculator
import json
import random
from ase.neighborlist import neighbor_list
import sys
import matplotlib as mpl
mpl.rcParams.update({'font.family': 'serif', 'font.size': 11})
from matplotlib import pyplot as plt
from ase import Atoms
from ase.data import covalent_radii


N = 12
TARGET = {
    'Si': 50,
    'Mg': 50
}

DB_NAME = "sqlite:///data/vacancy_sweep_no_periodic.db"
REF = 'ref_energies'
SOL = 'sol_pos'
VAC = 'vac_energy'
COMMENT = 'comment'

def create_precipitate(atoms, tags, radius_generator):
    pos = atoms.get_positions().copy()
    center = np.mean(pos, axis=0)
    pos -= center
    radii = np.sqrt(np.sum(pos[:, :2]**2, axis=1))
    indices = np.argsort(radii)
    assert len(center) == 3
    numSol = {
        'Si': 0,
        'Mg': 0
    }

    r = radius_generator()
    for layer in range(2, max(tags)-2):
        if layer%2 == 0:
            r = radius_generator()

        symbol = 'Mg' if layer%2 == 0 else 'Si'
        for i in indices:
            if radii[i] < r and numSol[symbol] < TARGET[symbol] and tags[i] == layer:
                atoms[i].symbol = symbol
                numSol[symbol] += 1
    
    count = sum(numSol.values())
    success = count == sum(TARGET.values())
    return atoms, success


def removeAl(atoms):
    for i in range(len(atoms)-1, -1, -1):
        if atoms[i].symbol == 'Al':
            del atoms[i]
    return atoms

def normal_radius():
    return np.abs(np.random.normal(loc=0.0, scale=3.0)) + 3.0

def mgsi(initial=None, comment=None):
    if initial is None:
        pureAl = bulk('Al', cubic=True)*(N, N, N)
        pureAl = wrap_and_sort_by_position(pureAl)
        tags, _ = get_layers(pureAl, (0, 0, 1))
        success = False
        while not success:
            print("Generating new initial precipitate")
            initial, success = create_precipitate(pureAl.copy(), tags, normal_radius)

    db = dataset.connect(DB_NAME)
    ref_tbl = db[REF]
    vac_tbl = db[VAC]
    sol_tbl = db[SOL]
    comment_tbl = db[COMMENT]

    runID = hex(random.randint(0, 2**32-1))
    if comment is not None:
        comment_tbl.insert({'runID': runID, 'comment': comment})

    eci = {}
    with open("data/almgsix_normal_ce.json", 'r') as infile:
        data = json.load(infile)
        eci = data['eci']

    settings = settingsFromJSON("data/settings_almgsiX_voldev.json")
    settings.basis_func_type = "binary_linear"
    atoms = attach_calculator(settings, initial.copy(), eci)
    atoms.numbers = initial.numbers
    ref_energy = atoms.get_potential_energy()
    ref_tbl.insert({'runID': runID, 'energy': ref_energy})

    for atom in atoms:
        if atom.symbol in ['Mg', 'Si']:
            pos = atom.position
            sol_tbl.insert({'runID': runID, 'symbol': atom.symbol, 'X': pos[0], 'Y': pos[1], 'Z': pos[2]})

    ref, neighbors = neighbor_list('ij', atoms, 3.0)
    for ref, nb in zip(ref, neighbors):
        if atoms[ref].symbol == 'Al' and atoms[nb].symbol in ['Mg', 'Si']:
            atoms[ref].symbol = 'X'
            e = atoms.get_potential_energy()
            atoms[ref].symbol = 'Al'
            pos = atoms[ref].position
            vac_tbl.insert({'runID': runID, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'energy': e})
    atoms = removeAl(atoms)
    
def show_energies(runID=None):
    db = dataset.connect(DB_NAME)
    vac_tab = db[VAC]
    ref_energies = db[REF]
    ref = {}
    for item in ref_energies.find():
        ref[item['runID']] =  item['energy']

    items = []
    if runID is None:
        items = vac_tab.find()

    energies = []
    for item in items:
        energies.append(item['energy'] - ref[item['runID']])

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(energies, 'o', mfc='none', color='black', alpha=0.05, markersize=2)
    ax.set_xlabel("Configuration no.")
    ax.set_ylabel("Energy (eV)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    plt.show()

def get_cluster(runID, tbl=None):
    if tbl is None:
        db = dataset.connect(DB_NAME)
        tbl = db[SOL]
    positions = []
    symbols = []
    for item in tbl.find(runID=runID):
        symbol = item['symbol']
        pos = [item['X'], item['Y'], item['Z']]
        positions.append(pos)
        symbols.append(symbol)
    return Atoms(positions=positions, symbols=symbols)

def neighbor_count(atoms, ref, min_dist, max_dist):
    pos = atoms.get_positions()
    pos -= ref
    dists = np.sqrt(np.sum(pos**2, axis=1)).tolist()
    count = {
        'Mg': 0,
        'Si': 0
    }
    for i, d in enumerate(dists):
        if d >= min_dist and d < max_dist:
            count[atoms[i].symbol] += 1
    return count

def extract_atoms(atoms, center, radius):
    pos = []
    symbols = []
    for a in atoms:
        diff = a.position - center
        r = np.sqrt(np.sum(diff**2))
        if r < radius:
            pos.append(a.position)
            symbols.append(a.symbol)
    return Atoms(positions=pos, symbols=symbols)


def explore_features():
    features = []
    energies = []
    db = dataset.connect(DB_NAME)
    vac_tbl = db[VAC]
    sol_tbl = db[SOL]
    ref_energies = db[REF]
    ref = {}
    for item in ref_energies.find():
        ref[item['runID']] =  item['energy']

    num_skipped = 0
    strage_structures = []
    for runID in ref.keys():
        print(f"Extracting ID {runID}")
        cluster = get_cluster(runID, tbl=sol_tbl)
        for item in vac_tbl.find(runID=runID):
            vac_pos = np.array([item['X'], item['Y'], item['Z']])
            count = neighbor_count(cluster, vac_pos, 2.5, 3.0)
            num_nn = sum(count.values())

            # Some vacancies wraps around, we just skip them here
            if num_nn > 0:
                energies.append(item['energy'] - ref[item['runID']])
                features.append([1.0, count['Si'], count['Mg']])
            else:
                num_skipped += 1

            if num_nn > 12:
                #from ase.build import stack
                vacancy = Atoms(positions=[vac_pos], symbols=['X'])
                atoms = cluster + vacancy
                strage_structures.append(atoms)

    print(f"Skipped {num_skipped} calculationes because no neighbours where found")
    energies = np.array(energies)
    features = np.array(features)
    coeff = np.linalg.lstsq(features, energies, rcond=None)[0]
    pred = features.dot(coeff)

    if strage_structures:
        view(strage_structures)

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(features[:, 1], energies, 'o', color='black', alpha=0.05, markersize=2)
    ax.set_xlabel("Number of Si atoms")
    ax.set_ylabel("Energy change (eV)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig('data/vacAnalysis/siDependency.png', dpi=300)

    fig2 = plt.figure(figsize=(4, 3))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(features[:, 2], energies, 'o', color='black', alpha=0.05, markersize=2)
    ax2.set_xlabel("Number of Mg atoms")
    ax2.set_ylabel("Energy change (eV)")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    fig2.tight_layout()
    fig2.savefig('data/vacAnalysis/mgDependency.png', dpi=300)
    plt.show()

def lowest_energy_change(num, include_special=True):
    db = dataset.connect(DB_NAME)
    ref_eng = {}
    for item in db[REF].find():
        ref_eng[item['runID']] = item['energy']

    specId = set()
    for item in db[COMMENT].find():
        if item['comment'].startswith("special"):
            specId.add(item['runID'])

    dE = {}
    rowIdRunId = {}
    for item in db[VAC].find():
        dE[item['id']] = item['energy'] - ref_eng[item['runID']]
        rowIdRunId[item['id']] = item['runID']

    tup = sorted([(e, i) for i, e in dE.items()])
    structures = []
    energies = []
    numInserted = 0
    for i in range(len(tup)):
        rowId = tup[i][1]
        runID = rowIdRunId[rowId]

        if runID in specId and not include_special:
            continue
        cluster = get_cluster(runID, tbl=db[SOL])
        item = db[VAC].find_one(id=rowId)
        vac = Atoms(positions=[[item['X'], item['Y'], item['Z']]], symbols=['X'])

        exists = False
        for e in energies:
            if abs(e - tup[i][0]) < 1e-4:
                exists = True
                break
        
        if exists:
            continue

        atoms = cluster + vac
        structures.append(atoms)
        energies.append(tup[i][0])
        numInserted += 1
        if numInserted == num:
            break
    print(energies)

    nns = []
    for s in structures:
        for a in s:
            if a.symbol == 'X':
                vpos = a.position
                break
        nns.append(extract_atoms(s, vpos, 3.0))
    
    combined = []
    for s, n in zip(structures, nns):
        combined.append(s)
        combined.append(n)
    view(combined)
    

def show_configs():
    db = dataset.connect(DB_NAME)
    vac_tbl = db[VAC]
    sol_tbl = db[SOL]

    # Find unique runIDs
    ids = sol_tbl.distinct('runID')
    
    images = []
    for runRes in ids:
        positions = []
        symbols = []
        charges = []
        runID = runRes['runID']
        for item in sol_tbl.find(runID=runID):
            symbols.append(item['symbol'])
            pos = [item['X'], item['Y'], item['Z']]
            positions.append(pos)
            charges.append(None)

        # Extract vacancy positions
        for item in vac_tbl.find(runID=runID):
            e = item['energy']
            charges.append(e)
            pos = [item['X'], item['Y'], item['Z']]
            positions.append(pos)
            symbols.append('X')

        max_charge = max([c for c in charges if c is not None])
        charges = [c - max_charge if c is not None else 0.0 for c in charges]
        images.append(Atoms(symbols=symbols, positions=positions, charges=charges))

    from ase.gui.images import Images
    from ase.gui.gui import GUI
    images = Images(images)
    images.covalent_radii[0] = covalent_radii[12]
    gui = GUI(images)
    gui.run()


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        mgsi()
    elif sys.argv[1] == 'energy':
        show_energies()
    elif sys.argv[1] == 'config':
        show_configs()
    elif sys.argv[1] == 'analyse':
        explore_features()
    elif sys.argv[1] == 'best':
        lowest_energy_change(100, include_special=True)
    elif sys.argv[1] == 'runspecial':
        special_file = 'data/specialMgSiClusters.traj'
        try:
            from ase.io.trajectory import TrajectoryReader
            traj = TrajectoryReader(special_file)
            for i, a in enumerate(traj):
                print(f"Running special structure: {i}")
                atoms = wrap_and_sort_by_position(a)
                comment = "special: Special structure for beta double prime precursors"
                mgsi(initial=atoms, comment=comment)
        except Exception as exc:
            print(exc)


