
import numpy as np
import pickle as pkl
import ase.units as units

class WangLandauSGC( object ):
    def __init__( self, atoms, calc, chemical_potentials={}, site_types=None, site_elements=None, Nbins=100, intial_f=2.71,
    flatness_criteria=0.8, fmin=1E-8, Emin=0.0, Emax=1.0 ):
        self.atoms = atoms
        self.site_types = site_types
        self.site_elements = site_elements
        self.chem_pot = chemical_potentials
        self.histogram = np.zeros(Nbins, dtype=np.int32)
        self.dos = np.zeros(Nbins)
        self.entropy = np.zeros(Nbins)
        self.Nbins = Nbins
        self.Emin = Emin
        self.Emax = Emax
        self.E = np.linspace(self.Emin, self.Emax, self.Nbins )
        self.f = initial_f
        self.flatness_criteria = flatness_criteria
        self.atoms_count = {}
        self.initialize()
        self.calc = calc
        self.current_bin = 0
        self.fmin = 1E-8

        if (( len(self.atoms) != len(self.site_types ) ):
            raise ValueError( "A site type for each site has to be specified!" )

        if ( not len(self.site_elements) == np.max(self.site_types)+1 ):
            raise ValueError( "Elements for each site type has to be specified!")

        # Check that a chemical potential have been given to all elements
        for site_elem in self.site_elements:
            for elem in site_elem:
                if ( not elem in self.chem_pot.keys() ):
                    raise ValueError("A chemical potential for {} was not specified".format(elem) )

    def initialize( self ):
        """
        Constructs the site elements and site types if they are not given
        """
        if ( self.site_types is None or self.site_elements is None ):
            self.site_types = [0 for _ in range(len(self.atoms))]
            symbols = []
            for atom in symbols:
                if ( not atom.symbol in symbols ):
                    symbols.append( atom.symbol )
            self.site_elements = [symbols]

        # Count number of elements
        for atom in self.atoms:
            if ( atom.symbol in self.atoms_count.keys() ):
                self.atoms_count[atom.symbol] += 1
            else:
                self.atoms_count[atom.symbol] = 1

    def get_bin( self, energy ):
        return int( (energy-self.Emin)*(self.Nbins-1)/(self.Emax-self.Emin) )

    def _step( self ):
        """
        Perform one MC step
        """
        indx = np.random.rand(len(atoms))
        symb = self.atoms[indx].symbol

        site_type = self.site_type[indx]
        possible_elements = self.site_elements[site_type]
        new_symbol = possible_elements[np.random.randint(low=0,high=len(possible_elements))]
        system_changes = [(indx,new_symbol)]
        self.calc.calculate( self.atoms, properties=["energy"], system_changes=system_changes )
        chem_pot_change = self.chem_pot[symbol]*(self.atoms_count[symbol]-1) + self.chem_pot[new_symbol]*(self.atoms_count[new_symbol]+1)
        energy = self.calc.results["energy"]-chem_pot_change
        selected_bin = self.get_bin(energy)


        bin_in_range = (selected_bin >= 0 ) and ( selected_bin < self.Nbins )
        rand_num = np.random.rand()
        accept_ratio = np.exp( self.entropy[self.current_bin]-self.entropy[selected_bin] )
        if ( rand_num < accept_ratio and bin_in_range ):
            self.current_bin = selected_bin
            self.atoms_count[symbol] -= 1
            self.atoms_count[new_symbol] += 1
            self.atoms[indx].symbol = new_symbol
        else:
            self.atoms[indx].symbol = symbol

        self.histogram[self.current_bin] += 1
        self.entropy[self.current_bin] += self.f

    def is_flat( self ):
        mean = np.mean( self.histogram )
        return np.min(self.histogram) > self.flatness_criteria*mean

    def save( self, fname ):
        with open(fname,'wb') as ofile:
            pkl.dump(self,ofile)

    def run( self, maxsteps=10000000 ):
        f_small_enough = False
        for i in range(maxsteps):
            self._step()
            if ( self.is_flat() ):
                self.histogram[:] = 0
                self.f *= 0.5

            if ( self.f < self.fmin ):
                f_small_enough = True
                break

        # Avoid overflow because large entropy, scale down by a factor so subtract of the mean
        # before exponentiating
        self.dos = np.exp(self.entropy - np.mean(self.entropy))
        #self.dos = np.exp(self.entropy)

    def sgc_partition_function( self, T ):
        """
        Computes the partition function in the SGC ensemble
        """
        return np.sum( self.dos*self._boltzmann_factor(T) )

    def _boltzmann_factor( self, T ):
        return np.exp( -self.E/(units.kB*T) )

    def sgc_avg_energy( self, T ):
        """
        Computes the average energy in the SGC ensemble
        """
        return np.sum( self.E*self.dos*self._boltzmann_factor )/self.sgc_partition_function(T)

    def sgc_heat_capacity( self, T ):
        """
        Computes the heat capacity in the SGC ensemble
        """
        e_mean = sgc_avg_energy(T)
        esq = np.sum(self.E**2 *self._boltzmann_factor(T) )/self.sgc_partition_function(T)
        return (esq-e_mean**2)/(units.kB*T**2)

    def sgc_potential( self, T ):
        """
        The thermodynamic potential in the SGC ensemble
        """
        return -units.kB*T*np.log(self.sgc_partition_function(T))
