import numpy as np
from matplotlib import pyplot as plt

FNAME = "/work/sophus/almgsi3d_strain_opposite/prec600K_trackvalues.csv"


def show():
    data = np.loadtxt(FNAME, delimiter=",")
    V = 64**3
    V_prec = 20**3
    iteration = data[:, 0]
    vol_energy = data[:, -1]
    surf_energy = data[:, -2]
    strain_energy = data[:, -4]

    vol_energy -= np.min(vol_energy)
    surf_energy -= np.min(surf_energy)
    strain_energy -= np.min(strain_energy)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(iteration, vol_energy)
    ax.plot(iteration, surf_energy)
    ax.plot(iteration, strain_energy)
    ax.plot(iteration, vol_energy + surf_energy + strain_energy)
    plt.show()


def aspect_ratio():
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from matplotlib import pyplot as plt
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName("/work/sophus/almgsi3d_strain_opposite/prec600K00000110000.vti")
    reader.Update()
    image = reader.GetOutput()
    rows, cols, depth = image.GetDimensions()
    sc = image.GetPointData().GetScalars()
    a = vtk_to_numpy(sc)
    conc = a[:, 0]
    conc = conc.reshape((rows, cols, depth))
    
    x = np.arange(0, rows)
    y = np.arange(0, cols)
    z = np.arange(0, depth)
    X, Y, Z = np.meshgrid(x, y, z)
    mask = conc > 0.1
    mask2 = conc > 0.2
    mask[mask2] = 0

    x_mask = mask*X
    y_mask = mask*Y
    z_mask = mask*Z
    x_min = np.min(x_mask[mask])
    x_max = np.max(x_mask[mask])
    y_min = np.min(y_mask[mask])
    y_max = np.max(y_mask[mask])
    z_min = np.min(z_mask[mask])
    z_max = np.max(z_mask[mask])
    print(x_min, x_max)
    print(y_min, y_max)
    print(z_min, z_max)

    aspect = (x_max - x_min)/(z_max - z_min)
    print(aspect)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mask[:, :, 35])
    plt.show()


if __name__ == '__main__':
    aspect_ratio()