import numpy as np
from matplotlib import pyplot as plt

FOLDER = "/home/davidkleiven/Documents/Dump/AdaptivePF/AdaptiveSingle/"
N = 256

dx4 = "DX4/ch_eta1_67.bin"
dx3 = "DX3/ch_eta1_49.bin"
dx2 = "DX2/ch_eta1_59.bin"
dx1 = "DX1/ch_eta1_3.bin"
def main():
    data = np.fromfile(FOLDER + dx1, dtype=">f8")

    data = data.reshape((N, N))
    lineX = data[int(N/2), :]
    lineY = data[:, int(N/2)]
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    for i in range(len(lineX)-1):
        if lineX[i] < 0.5 and lineX[i+1] > 0.5:
            xmin = i
        if lineX[i] > 0.5 and lineX[i+1] < 0.5:
            xmax = i
        if lineY[i] < 0.5 and lineY[i+1] > 0.5:
            ymin = i
        if lineY[i] > 0.5 and lineY[i+1] < 0.5:
            ymax = i
        
    b = xmax - xmin
    a = ymax - ymin
    rho = abs(b - a)/abs(b+a)
    print(xmin, xmax, ymin, ymax, rho)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(data)
    plt.show()

if __name__ == '__main__':
    main()