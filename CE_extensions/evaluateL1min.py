from ase.ce.evaluate import Evaluate # This requires the ase repo with CE developed by Jin Chang at DTU
from sklearn import linear_model
from scipy.optimize import minimize
import numpy as np
import copy
from matplotlib import pyplot as plt

class EvaluateL1min( Evaluate ):
    def __init__( self, db_name, cluster_names=None, lamb=0.0, eci=None, threshold=0.01, alpha=0.01):
        """
        Class that computes the ECIs based on minimizing the l1 norm of the ECI vector

        Parameters
        ----------
        See the evaluate class

        threshold: ECIs below threshold*max(|{eci}|) are set to zero
        alpha: Regularization parameter in sklearn.linear_model.Lasso
        """
        self.threshold = threshold
        self.alpha = alpha
        Evaluate.__init__( self, db_name, cluster_names=cluster_names, lamb=lamb, eci=eci )
        self.selected_cluster_names = []
        self.selected_features = None
        self.find_optimal_alpha()

    def get_eci( self ):
        """
        Compute the ECIs based on DFT data.
        Overrides the parents get_eci method
        """
        #if ( self.eci is not None ):
        #    return self.eci

        n_col = self.cf_matrix.shape[1]

        linmod = linear_model.Lasso( alpha=self.alpha, copy_X=True, fit_intercept=False )
        linmod.fit( self.cf_matrix, self.e_dft )
        self.eci = linmod.coef_
        support = self.select_eci()
        reduced_matrix = self.cf_matrix[:,support]

        # Perform a new fit where only the selected ECIs are included
        linmod.fit( reduced_matrix, self.e_dft )
        self.eci = np.zeros( n_col )
        self.eci[self.selected_features] = linmod.coef_
        return self.eci

    def get_eci_loo( self, leave_out ):
        """
        Computes the ECI by leaving one out

        Parameter
        ---------
        leave_out: Index of the measurement to leave out
        """
        edft_copy = copy.deepcopy( self.e_dft )
        cfm_copy = copy.deepcopy( self.cf_matrix )
        self.e_dft = np.delete( self.e_dft, leave_out )
        self.cf_matrix = np.delete( self.cf_matrix, leave_out, 0 )
        eci_loo = self.get_eci()

        # Copy back the full version
        self.cf_matrix = cfm_copy
        self.e_dft = edft_copy

        # Update the ECIs to the default
        self.get_eci()
        return eci_loo


    def _eci_from_solution( self, solution ):
        """
        Extracts the ECIs from the solution vector

        Parameters
        ----------
        solution: The solution vector from the linear program
        """
        number_of_eci = self.cf_matrix.shape[1]
        eci = np.zeros( number_of_eci )
        for i in range( number_of_eci ):
            if ( solution[i] == 0.0  ):
                eci[i] = -solution[i+number_of_eci]
            elif ( solution[i+number_of_eci] == 0.0 ):
                eci[i] = solution[i]
            else:
                print (solution)
                raise ValueError("The solution indicates that the ECI is both positive and negative. This should never happen.")
        return eci

    def select_eci( self, threshold=None ):
        """
        Selects clusters that contributes most
        """
        if ( not threshold is None ):
            self.threshold = threshold

        max_eci = np.max( np.abs(self.eci) )
        number_of_clusters_contributing = 0
        self.selected_features = np.abs(self.eci) > self.threshold*np.median(np.abs(self.eci))
        self.selected_cluster_names = [name for name,boolean in zip(self.cluster_names,self.selected_features) if (boolean)]
        return self.selected_features


    def find_optimal_alpha( self ):
        """
        Find the optimal value for the regularization parameter alpha by minimizing the CV score
        """
        res = minimize( minimzation_target, np.log10(self.alpha), args=(self,), method="Nelder-Mead" )
        if ( not res["success"] ):
            raise RuntimeError( res["message"] )
        print ("Optimal penalization value (alpha): %.2E. Minimum CV score: %.2E"%(10**res["x"],res["fun"]))

    def plot_selected_eci( self ):
        if ( self.selected_cluster_names == [] ):
            print ("No clusters have been selected")

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        selected_eci = self.eci[np.abs(self.eci)>0.0]
        x = np.arange(len(selected_eci))
        ax.bar( x, selected_eci )
        ax.set_xticks(x)
        ax.set_xticklabels( self.selected_cluster_names, rotation=45, ha="center" )
        ax.set_xlabel( "Cluster name")
        ax.set_ylabel( "Effective Cluster Interaction (eV/atom)")
        plt.show()

def minimzation_target( log10alpha, obj ):
    """
    Cost function to minimize in order to find the optimal value of alpha
    """
    obj.alpha = 10**log10alpha
    cv = obj.cv_loo()
    return cv
