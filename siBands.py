import ase
from ase import build
import gpaw as gp
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt

def main():
    si = build.bulk( "Si", "diamond", 5.43 )
    calc = gp.GPAW( mode=gp.PW(200), xc="PBE", kpts=(8,8,8), random=True,
    occupations=gp.FermiDirac(0.01), txt="Si_GS.txt")
    si.calc = calc
    si.get_potential_energy()
    calc.write( "Si_GS.gpw" )

    # Restart based on the previous simulation
    calc = gp.GPAW( "Si_GS.gpw", nbands=16, fixdensity=True, symmetry="off",
    kpts={"path":"GXWKL","npoints":60},
    convergence={"bands":8})

    calc.get_potential_energy()
    bs = calc.band_structure()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    bs.plot( ax=ax, emax=10.0, show=False )
    ax.set_ylabel( "Energy (ev)" )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    for line in ax.get_lines():
        line.set_color("black")
    plt.show()


if __name__ == "__main__":
    main()
