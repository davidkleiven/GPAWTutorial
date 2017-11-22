import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
#mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt

class ECIPlotter( object ):
    def __init__( self, eci ):
        self.eci = eci

    def plot( self, tight=False ):
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
        if ( tight ):
            plt.tight_layout()
        return fig
