import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
#mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt

class ECIPlotter( object ):
    def __init__( self, eci ):
        self.eci = eci

    def sort_on_cluster_size( self ):
        """
        Sort the atoms according to cluster size
        """
        cluster_size = []
        eci_names = {}
        eci_values = {}

        for key,value in self.eci.iteritems():
            size = int(key[1])
            if ( size in eci_names.keys() ):
                eci_names[size].append(key)
                eci_values[size].append(value)
            else:
                eci_names[size] = [key]
                eci_values[size] = [value]

        # Within each size sort on absolute value
        for key,value in eci_values.iteritems():
            indx_srt = np.argsort(np.abs(value))[::-1]
            new_list = [value[indx] for indx in indx_srt]
            eci_values[key] = new_list
            new_list = [eci_names[key][indx] for indx in indx_srt]
            eci_names[key] = new_list
        return eci_names, eci_values

    def plot( self, tight=False, show_names=False ):

        labels = {
            0:"Bias",
            1:"Singlets",
            2:"Doublets",
            3:"Triplets",
            4:"Quadruplets"
        }
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        eci_names, eci_vals = self.sort_on_cluster_size()
        prev = 0
        for key,value in eci_vals.iteritems():
            indx = np.arange(prev,prev+len(value) )
            ax.bar( indx, value, label=labels[key] )
            prev = indx[-1]+1
        ax.legend( loc="best", frameon=False )
        ax.axhline(0.0, color="black", linewidth=0.5, ls="--")
        ax.set_xticklabels([])
        ax.set_ylabel( "ECI (eV/atom)" )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ( show_names ):
            ax.set_xticklabels(keys, rotation="vertical" )
        return fig
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ind = range(len(self.eci.keys()))
        eci_vals = [value for key,value in self.eci.iteritems()]
        sorted_arg = np.argsort(np.abs(eci_vals))[::-1]
        sorted_eci = [eci_vals[indx] for indx in sorted_arg]
        keys = [self.eci.keys()[indx] for indx in sorted_arg]

        # Reformat the keys
        formatted_keys = []
        for key in keys:
            split = key.split("_")
            new_key = "\$"+split[0]+"_{"
            for i in range(1,len(split)):
                new_key += split[i]+"."
            new_key += "}\$"
            formatted_keys.append(new_key)
        keys = formatted_keys
        eci_vals = sorted_eci
        ax.bar( ind, eci_vals )
        ax.set_xlabel("Cluster name")
        ax.set_ylabel("ECI (eV/atom)" )
        ax.set_xticklabels(keys, rotation="vertical" )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xticks(ind)
        ax.axhline(0.0, color="black", linewidth=0.5, ls="--")
        """
        if ( tight ):
            plt.tight_layout()
        return fig
