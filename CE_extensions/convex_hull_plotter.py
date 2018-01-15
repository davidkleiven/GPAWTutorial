import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
#mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt
from ase.db import connect
from scipy.spatial import ConvexHull
from scipy.stats import linregress

class QHull( object ):
    def __init__( self, db_name ):
        self.db = connect( db_name )
        self.energies = []
        self.compositions = []
        self.formulas = []

    def concentration( self, symbs, natoms ):
        """
        Create dictionary with the concentrations
        """
        concs = {}
        for symb in symbs:
            if ( not symb in concs.keys() ):
                concs[symb] = 1
            else:
                concs[symb] += 1
        for key in concs.keys():
            concs[key] /= float(natoms)
        return concs

    def get_energies_from_relaxed_structures( self ):
        """
        Returns the energy of all relaxed structures
        """
        for row in self.db.select( converged=1 ):
            N = row.natoms
            self.energies.append( row.energy/N )
            symbs = row.symbols
            self.compositions.append( self.concentration(symbs,N) )
            self.formulas.append( row.formula )
        return self.compositions, self.energies

    def reference_energies( self ):
        """
        Computes the reference energies
        """
        ref_eng = {}
        for entry in self.compositions:
            for key,value in entry.iteritems():
                if ( key in ref_eng.keys() ):
                    continue
                if ( value == 1.0 ):
                    ref_eng[key] = value

    def plot( self, el1 ):
        """
        Creates a plot of the convex hull of all converged simulations
        el1 is positioned on the left and el2 is positioned on the left
        """

        comps, energies = self.get_energies_from_relaxed_structures()
        comp_list = [entry[el1] for entry in comps]

        comp_list = np.array( comp_list )
        energies = np.array( energies )

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( comp_list, energies, 'o', mfc="none" )

        slope, interscept, rvalue, pvalue, stderr = linregress( comp_list, energies )
        energies -= (interscept + comp_list*slope )
        points = np.vstack((comp_list,energies)).T
        hull = ConvexHull( points )

        for simplex in hull.simplices:
            ax.plot( points[simplex,0], points[simplex,1], "k-")
        ax.set_xlabel( "Concentration {}".format(el1) )
        ax.set_ylabel( "Formation energy (eV/atom)" )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        return fig
