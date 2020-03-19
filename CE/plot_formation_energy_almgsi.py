import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'axes.unicode_minus': False, 'font.size': 18, 'svg.fonttype': 'none'})
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
import ternary
from skimage import measure


ref_al = -3.737
ref_mg = -1.599
ref_si = -4.864


class GaussianBlur(object):
    def __init__(self, x, y, z, data, r=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.data = data
        self.r = r
        self.eucl_x = self.x + 0.5*self.y
        self.eucl_y = np.sqrt(3)/2*self.y

    def evaluate(self, p):
        if p[1] > 0.505:
            return 10.0
        eucledian = p[0]*np.array([1, 0]) + p[1]*np.array([0.5, np.sqrt(3)/2])
        wx = np.exp(-((self.eucl_x-eucledian[0])/self.r)**2)
        wy = np.exp(-((self.eucl_y-eucledian[1])/self.r)**2)
        return np.sum(self.data*wx*wy)/np.sum(wx*wy)


def formation_energy():
    fname = "data/almgsi_ce.csv"

    data = np.loadtxt(fname, delimiter=',')
    al_conc = data[:, 0]
    mg_conc = data[:, 1]
    si_conc = data[:, 2]
    e_pred = data[:, 3]
    e_dft = data[:, 4]

    e_form_dft = e_dft - al_conc*ref_al - mg_conc*ref_mg - si_conc*ref_si
    e_form_ce = e_pred - al_conc*ref_al - mg_conc*ref_mg - si_conc*ref_si
    return al_conc, mg_conc, si_conc, e_form_dft, e_form_ce


def ternay_plot_dft():
    al_conc, mg_conc, si_conc, e_form_dft, e_form_ce = formation_energy()
    # Add pure Mg and Al
    e_form_dft = np.append(e_form_dft, [0.0, 0.0])
    e_form_ce = np.append(e_form_ce, [0.0, 0.0])
    al_conc = np.append(al_conc, [0.0, 1.0])
    si_conc = np.append(si_conc, [0.0, 0.0])
    mg_conc = np.append(mg_conc, [1.0, 0.0])

    cnv_hull = ConvexHull(np.vstack((mg_conc, si_conc, e_form_dft)).T)
    unique_index = set()
    for simpl in cnv_hull.simplices:
        for s in simpl:
            unique_index.add(s)

    # Add pure Al
    averager = GaussianBlur(al_conc, si_conc, mg_conc, e_form_dft, r=0.1)

    scale = 60
    fig, tax = ternary.figure(scale=scale)
    points = [(al_conc[i]*scale, si_conc[i]*scale, mg_conc[i]*scale) for i in range(len(al_conc))]
    tax.boundary(linewidth=1.0)
    #tax.ticks(axis='lbr', linewidth=1, multiple=10)
    tax.heatmapf(averager.evaluate, cmap='terrain', style="triangular", scale=scale, vmax=0.05, boundary=True, 
                 cbarlabel='Formation energy (eV/atom)', scientific=False)
    tax.scatter(points, marker='x', zorder=5, color='#777777')
    #tax.right_axis_label('Si concentration')
    #tax.left_axis_label('Mg concentration')
    #tax.bottom_axis_label('Al concentration')
    tax.gridlines(multiple=5, color='#555555')
    tax.clear_matplotlib_ticks()

    # Plot points on convex hull
    cnv_hull_points = [(al_conc[i]*scale, si_conc[i]*scale, mg_conc[i]*scale) for i in unique_index]
    print([(al_conc[i], si_conc[i], mg_conc[i]) for i in unique_index])
    #tax.scatter(cnv_hull_points, marker='o', zorder=10, facecolors='#9ecae1', edgecolors='#777777', s=80)
    tax.savefig('data/ternary_plot_pixels.png', dpi=300)
    tax.show()

def main():
    al_conc, mg_conc, si_conc, e_form_dft, e_form_ce = formation_energy()

    # Add pure Mg
    e_form_dft = np.append(e_form_dft, [0.0])
    e_form_ce = np.append(e_form_ce, [0.0])
    al_conc = np.append(al_conc, [0.0])
    si_conc = np.append(si_conc, [0.0])
    mg_conc = np.append(mg_conc, [1.0])

    mask = mg_conc < 0.51
    mask[mg_conc < 0.48] = 0
    indx = np.argmin(e_form_dft[mask])
    print(si_conc[mask][indx])
    print(al_conc[mask][indx])
    print(mg_conc[mask][indx])

    fig = plt.figure()
    x = mg_conc/(si_conc + mg_conc)
    x = mg_conc
    ax = fig.add_subplot(1, 1, 1)

    points_dft = np.vstack((x, e_form_dft)).T
    hull_dft = ConvexHull(points_dft)

    for simplex in hull_dft.simplices:
        x_smp = points_dft[simplex, 0]
        y_smp = points_dft[simplex, 1]
        if np.any(y_smp > 0.0):
            continue
        if np.abs(x_smp[0] - x_smp[1]) < 1E-3:
            continue
        ax.plot(x_smp, y_smp, color="black", lw=3)

    ax.plot(x, e_form_ce, 's', color='#7395ae', mfc='none', label="CE",
            markeredgewidth=1.5)
    ax.plot(x, e_form_dft, 'o', color='#5d5c61', mfc='none', label="DFT",
            markeredgewidth=1.5)
    ax.set_xlabel("Mg concentration")
    ax.set_ylabel("Formation energy (eV/atom)")
    ax.legend(loc="best", frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

if __name__ == '__main__':
    #main()
    ternay_plot_dft()
