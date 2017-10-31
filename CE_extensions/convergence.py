import sys
sys.path.insert(1,"/home/davidkl/Documents/aseJin")
from ase.ce.evaluate import Evaluate
import copy
import numpy as np
from matplotlib import pyplot as plt


class ConvergenceCheck(object):
    def __init__( self, evaluator ):
        if ( not isinstance(evaluator,Evaluate) ):
            raise TypeError( "Argument evaluator has to be of type Evaluate" )
        self.evaluator = evaluator
        self.cv_score_N = []
        self.cv_score_N_Nprobe = []

    def converged( self ):
        """
        Checks whether the CV score when adding probe structures are smaller than
        the CV score of the initial data
        """
        cf_matrix = copy.deepcopy(self.evaluator.cf_matrix)
        e_dft = copy.deepcopy(self.evaluator.e_dft)
        gens = self.evaluator.generations
        result = False
        for maxgen in range(np.min(self.evaluator.generations),np.max(self.evaluator.generations)):
            self.evaluator.cf_matrix = cf_matrix[gens<=maxgen,:]
            self.evaluator.e_dft = e_dft[gens<=maxgen]

            # Compute the ECI
            self.evaluator.get_eci()
            self.cv_score_N.append( self.evaluator.cv_loo() )

            # Add the probe structures
            probe_struct = cf_matrix[gens==maxgen+1,:]
            probe_e_dft = e_dft[gens==maxgen+1]
            e_pred = probe_struct.dot( self.evaluator.eci )
            cv_N_Nprobe = np.sqrt( np.sum( (e_pred-probe_e_dft)**2 )/len(probe_e_dft) )

            # Do not update the ECIs
            self.cv_score_N_Nprobe.append( self.evaluator.cv_loo() )

            if ( self.cv_score_N_Nprobe <= self.cv_score_N ):
                result = True
        # Set the original matrix back into the evaluator
        self.evaluator.cf_matrix = cf_matrix
        self.evaluator.e_dft = e_dft
        return result

    def plot_energy_with_gen_info( self ):
        """
        Plots the energy from DFT and CE. Color codes indicates generation
        """
        gens = self.evaluator.generations
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        e_dft = self.evaluator.e_dft
        ax.plot( e_dft, e_dft )
        for gen in range(np.min(self.evaluator.generations),np.max(self.evaluator.generations)+1):
            e_dft = self.evaluator.e_dft[gens==gen]
            e_pred = self.evaluator.e_pred[gens==gen]
            ax.plot( e_pred, e_dft, 'o', mfc="none", markeredgewidth=2, label="G%d"%(gen))
        ax.set_xlabel("$E_{CE}$ (eV/atom)")
        ax.set_ylabel("$E_{DFT}$ (eV/atom)")
        ax.legend( loc="best" )
        return fig

    def plot_cv_score( self ):
        """
        Plots the cross validation score
        """
        if ( self.cv_score_N == [] ):
            self.converged()
        gens = np.arange( np.max(self.evaluator.generations) )
        print (gens,self.cv_score_N)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( gens, self.cv_score_N, label="CV(N)" )
        ax.plot( gens, self.cv_score_N_Nprobe, label="CV(N+N_{probe})")
        ax.set_xlabel( "Generation" )
        ax.set_ylabel( "Cross Validation Score" )
        ax.legend( loc="best", frameon=False, labelspacing=0.05 )
        return fig
