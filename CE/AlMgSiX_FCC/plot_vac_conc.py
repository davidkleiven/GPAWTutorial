import matplotlib as mpl
mpl.rcParams.update({'font.family': 'serif', 'font.size': 11, 'svg.fonttype': 'none'})
from matplotlib import pyplot as plt
import sqlite3
import numpy as np
from scipy.integrate import cumtrapz
from scipy.stats import linregress
import pandas as pd
from ase.units import kB

db_name = "data/almgsi_mc_sgc.db"
N = 32

def plot_single_temp():
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    sql = "SELECT occupLayer0,occupLayer1,occupLayer2,occupLayer3,occupLayer4,occupLayer5,occupLayer6,occupLayer7 "
    sql += "FROM local_environ_mgsi WHERE temperature=300 AND mu_c1_2>? AND mu_c1_2<? "
    sql += "AND initial='data/mgsi_active_matrix.xyz'"
    chem_pots = [4.1, 4.15, 4.2]
    colors = ['#1c1c14', '#742d18', '#453524']

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    for i, mu in enumerate(chem_pots):
        cur.execute(sql, (mu-0.001,mu+0.001))
        res = []
        for item in cur.fetchall():
            res.append(np.array(item)/N)
        avg = np.mean(res, axis=0)
        x = list(range(len(avg)))
        ax.plot(x, avg, marker='o', label=f"{mu}", mfc='none', color=colors[i%len(colors)])
        ax.set_yscale('log')
    con.close()
    ax.legend(frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Distance from interface")
    ax.set_ylabel("Vac. concentration")
    fig.tight_layout()
    plt.show()

def plot_different_interfaces():
    occupLayer = ','.join(f"occupLayer{i}" for i in range(7))
    sql = f"SELECT {occupLayer} FROM local_environ_mgsi WHERE "
    sql += "mu_c1_2 > 4.14 AND mu_c1_2 < 4.16 AND initial=? and temperature=300"
    initial = ["data/mgsi_active_matrix.xyz",
               "data/mgsi_active_matrix_magnesium_interface.xyz",
               "data/mgsi_active_matrix_silicon_interface.xyz"]
    names = ["Alt.", "Mg", "Si"]
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    colors = ['#1c1c14', '#742d18', '#453524']
    for i, init in enumerate(initial):
        cur.execute(sql, (init,))
        res = []
        for item in cur.fetchall():
            res.append(np.array(item)/N)
        avg = np.mean(res, axis=0)
        ax.plot(avg, marker='o', mfc='none', label=names[i], color=colors[i])
    con.close()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Distance from interface")
    ax.set_ylabel("Interface coverage")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

def excess_interface_energy(chem_pot, avg_interface, avg_bulk):
    avg_interface = np.array(avg_interface)
    avg_bulk = np.array(avg_bulk)
    integrand = (1.0-avg_interface)*avg_bulk/(1.0 - avg_bulk) - avg_interface
    res = cumtrapz(integrand, x=chem_pot, initial=0.0)
    return res

def plot_first_layer_excess(T):
    sql = f"SELECT occupLayer0,occupLayer7 FROM local_environ_mgsi WHERE "
    sql += f"mu_c1_2 > ? AND mu_c1_2 < ? AND initial=? and temperature={int(T)}"
    initial = ["data/mgsi_active_matrix.xyz",
               "data/mgsi_active_matrix_magnesium_interface.xyz",
               "data/mgsi_active_matrix_silicon_interface.xyz"]
    names = ["Alt.", "Mg", "Si"]
    fig = plt.figure(figsize=(4, 3))
    fig2 = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax2 = fig2.add_subplot(1, 1, 1)
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    colors = ['#1c1c14', '#742d18', '#453524']
    chem_pots = [4.025, 4.05, 4.075, 4.1, 4.125, 4.15, 4.175]
    tol = 1e-4
    for i, init in enumerate(initial):
        excess = []
        bulk_conc = []
        for c in chem_pots:
            cur.execute(sql, (c-tol, c+tol, init))
            avg = 0.0
            avg_matrix = 0.0
            num = 0
            for item in cur.fetchall():
                avg += item[0]
                avg_matrix += item[1]
                num += 1
            avg /= (N*num)
            avg_matrix /= (N*num)
            excess.append(avg)
            bulk_conc.append(avg_matrix)
        excess_energy = excess_interface_energy(chem_pots, excess, bulk_conc)
        ax.plot(chem_pots, excess, marker='o', mfc='none', label=names[i], color=colors[i])
        slope, interscept, _, _, _ = linregress(excess, excess_energy)
        print(f"Interface: {names[i]} slope {slope}")
        ax2.plot(excess, excess_energy, 'o', mfc='none', color=colors[i])
        x = np.array([0.0, 0.3])
        ax2.plot(x, interscept + slope*x, ls='--', color=colors[i], label=names[i])
    con.close()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Chemical potential (eV/atom)")
    ax.set_ylabel("Interface coverage")
    ax.legend(frameon=False)
    fig.tight_layout()

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlabel("Interface coverage")
    ax2.set_ylabel("Change in interfacial energy (eV/atom)")
    ax2.legend(frameon=False)
    fig2.tight_layout()
    plt.show()

def plot_binding_energy_vs_temp():
    data = pd.read_csv("data/binding_energies.csv")
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    unique_interf = set(data['InterfaceType'])
    datasets = {k: {'T': [], 'E': []} for k in unique_interf}
    colors = ['#1c1c14', '#742d18', '#453524']

    for _, row in data.iterrows():
        datasets[row['InterfaceType']]['T'].append(row['Temperture'])
        datasets[row['InterfaceType']]['E'].append(row['BindingEnergy'])
    
    for i, interface in enumerate(unique_interf):
        E = np.array(datasets[interface]['E'])
        T = np.array(datasets[interface]['T'])
        normalized = E/(kB*T)
        ax.plot(T, normalized, label=interface, marker='o', mfc='none', color=colors[i])
    ax.legend(frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('$\\epsilon_V/kT$')
    fig.tight_layout()
    plt.show()



plot_single_temp()
plot_different_interfaces()
plot_first_layer_excess(300)
plot_binding_energy_vs_temp()
