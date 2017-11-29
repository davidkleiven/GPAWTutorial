from ase import units
import numpy as np
from matplotlib import pyplot as plt

class WangLandauSGCAnalyzer( object ):
    def __init__( self, energy, dos, chem_pot ):
        """
        Object for analyzing thermodynamics from the Density of States in the
        Semi Grand Cannonical Ensemble
        """
        self.E = energy
        self.dos = dos
        self.E0 = np.min(self.E)
        self.chem_pot = chem_pot

    def partition_function( self, T ):
        """
        Computes the partition function in the SGC ensemble
        """
        return np.sum( self.dos*self._boltzmann_factor(T) )

    def _boltzmann_factor( self, T ):
        """
        Returns the boltzmann factor
        """
        return np.exp( -(self.E-self.E0)/(units.kB*T) )

    def internal_energy( self, T ):
        """
        Computes the average energy in the SGC ensemble
        """
        return np.sum( self.E*self.dos*self._boltzmann_factor(T) )/self.partition_function(T)

    def heat_capacity( self, T ):
        """
        Computes the heat capacity in the SGC ensemble
        """
        e_mean = internal_energy()
        esq = np.sum(self.E**2 *self._boltzmann_factor(T) )/self.partition_function(T)
        return (esq-e_mean**2)/(units.kB*T**2)

    def free_energy( self, T ):
        """
        The thermodynamic potential in the SGC ensemble
        """
        return -units.kB*T*np.log(self.partition_function(T)) + self.E0

    def plot_dos( self ):
        """
        Plots the density of states
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( self.E, self.dos, ls="steps" )
        ax.set_yscale("log")
        ax.set_xlabel( "Energy (eV/atom)" )
        ax.set_ylabel( "Density of states" )
        return fig
