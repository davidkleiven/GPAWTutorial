import numpy as np
from matplotlib import pyplot as plt

def energies(noise):
    x = np.linspace(0.0, 10.0, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    E = 0.5*x**2
    E_model = E + np.random.normal(0.0, scale=noise, size=len(E))
    ax.plot(x, E_model)
    ax.plot(x, E)
    return E, E_model
    
def dos():
    E, E_model = energies(2.0)
    dos1, edges1 = np.histogram(E, bins=100)
    dos2, edges2 = np.histogram(E_model, bins=60)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(edges1[1:], dos1, drawstyle='steps')
    ax.plot(edges2[1:], dos2, drawstyle='steps')
    kB = 0.086
    # Free energy
    T = np.linspace(10.0, 500, 100)
    Z1 = np.zeros_like(T)
    Z2 = np.zeros_like(T)
    E1 = np.zeros_like(T)
    E2 = np.zeros_like(T)
    C1 = np.zeros_like(T)
    C2 = np.zeros_like(T)
    dE1 = edges1[1:] - np.min(edges1[1:])
    dE2 = edges2[1:] - np.min(edges2[1:])
    for i in range(len(T)):
        Z1[i] = np.sum(dos1*np.exp(-dE1/(kB*T[i])))
        Z2[i] = np.sum(dos2*np.exp(-dE2/(kB*T[i])))

        E1[i] = np.sum(dE1*dos1*np.exp(-dE1/(kB*T[i])))
        E2[i] = np.sum(dE2*dos2*np.exp(-dE2/(kB*T[i])))

        C1[i] = np.sum(dE1**2*dos1*np.exp(-dE1/(kB*T[i])))/(kB*T[i]**2)
        C2[i] = np.sum(dE2**2*dos2*np.exp(-dE2/(kB*T[i])))/(kB*T[i]**2)
    
    figF = plt.figure()
    axF = figF.add_subplot(1, 1, 1)
    axF.plot(T, E1)
    axF.plot(T, E2)
    axCv = axF.twinx()
    axCv.plot(T, C1)
    axCv.plot(T, C2)


#energies()
dos()
plt.show()
