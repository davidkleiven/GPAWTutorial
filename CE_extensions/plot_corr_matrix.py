import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
#mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt

class CovariancePlot( object ):
    def __init__( self, evaluator, constant_term_column=None ):
        self.cf_mat = evaluator.cf_matrix

        if ( not constant_term_column is None ):
            self.cf_mat = np.delete( self.cf_mat, constant_term_column, axis=1 )

        indx = []
        for i in range(0,self.cf_mat.shape[0]):
            if ( np.std(self.cf_mat[i,:]) == 0 ):
                indx.append(i)
        self.cf_mat = np.delete( self.cf_mat, indx, axis=0 )
        self.correlation_matrix = np.corrcoef( self.cf_mat )

    def plot( self ):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow( self.correlation_matrix, interpolation="none", cmap="coolwarm")
        cbar = fig.colorbar(im)
        cbar.set_label( "Structure correlation" )
        ax.set_xlabel( "Structure number" )
        ax.set_ylabel( "Structure number" )
        return fig
