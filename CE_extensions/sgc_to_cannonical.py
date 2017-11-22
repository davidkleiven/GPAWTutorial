from scipy.stats import linregress
from scipy import interpolate

class SGCToCanonicalConverter(object):
    def __init__( self, wl_simulations ):
        self.wl_simulations = wl_simulations
        self.n_atoms = len(self.wl_simulations[0].atoms) # Should be the same in all WLs
        self.composition = None
        self.chemical_potentials = None
        self.sgc_potentials = None

    def get_composition_one_element( self, T, element, spline_order=3, n_chem_pots=50 ):
        """
        Computes the compositions
        """
        chem_pots = []
        thermo_potentials = []
        for wl in self.wl_simulations:
            chem_pots.append(wl.chem_pot[element])
            thermo_potentials.append( wl.sgc_potential(T) )

        # Sort the chemical potentials
        sort_arg = np.argsort(chem_pots)
        sorted_chem_pots = [chem_pots[indx] for indx in sort_arg]
        sorted_thermo_pots = [thermo_potentials[indx] for indx in sort_arg]
        interpolator = interpolate.interp1d(sorted_chem_pots,sorted_thermo_pots,k=spline_order)
        new_chem_pots = np.linsapace(np.min(sorted_chem_pots), np.max(sorted_chem_pots), n_chem_pots)
        x = interpolate.splev(new_chem_pots,interpolator,der=1)/self.n_atoms
        phi = interpolate.splev(new_chem_pots)
        return new_chem_pots, x, phi

    def get_compositions( self, T, spline_order=3, n_chem_pots=50 ):
        """
        Get the composition of all elements.
        NOTE: One of the elements will have a composition of zero since its
        chemical potential was used as a reference.
        The proper composition of this element is 1 minus the composition
        of all the others
        """
        elms = self.wl_simulations.atoms_count.keys()
        comp = {}
        sgc_pod = {}
        chem_pot = {}
        for symbol in elms:
            chem_pot[symbol], comp[symbol], sgc_pot[symbol] = self.get_composition_one_element( T, symbol, spline_order=spline_order )
        self.composition = comp
        self.sgc_potentials = sgc_pot
        self.chemical_potentials = chem_pot
        return chem_pot, comp, sgc_pot

    def free_energy( self, T, composition ):
        """
        Return the Helmholtz free energy
        """
        if ( self.composition is None ):
            self.get_compositions()

        thermo_potentials = []
        for wl in self.wl_simulations:
            thermo_potentials.append( wl.sgc_potential(T) )
            for key,value in self.compositions:
                interpolator = interpolator.interp1d( value, self.chemical_potentials[key] )
                thermo_potentials[-1] += interpolator(composition[key])*composition[key]
        # All the SGC potentials + the contribution to from chemical potentials
        # should be the same, but use the mean to reduce inaccuracies due to numerical errors, inaccuracies in the DOS etc.
        free_energy = np.mean( thermo_potentials )
        return free_energy
