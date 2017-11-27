
import numpy as np
import pickle as pkl
import ase.units as units
from matplotlib import pyplot as plt
from scipy import interpolate
import copy
import json
import sqlite3 as sq
import io

class WangLandauSGC( object ):
    def __init__( self, atoms, calc, db_name, db_id, site_types=None, site_elements=None, Nbins=100, initial_f=2.71,
    flatness_criteria=0.8, fmin=1E-8, Emin=None, Emax=None ):
        self.atoms = atoms
        self.site_types = site_types
        self.site_elements = site_elements
        self.possible_swaps = []
        self.chem_pot = chemical_potentials
        self.histogram = np.zeros(Nbins, dtype=np.int32)
        self.dos = np.zeros(Nbins)
        self.entropy = np.zeros(Nbins)
        self.Nbins = Nbins
        self.Emin = 0.0
        self.Emax = 1.0
        self.update_bounds_every = 100
        self.E = np.linspace(self.Emin, self.Emax, self.Nbins )
        self.f = initial_f
        self.f0 = initial_f
        self.flatness_criteria = flatness_criteria
        self.atoms_count = {}
        self.initialize()
        self.calc = calc
        self.current_bin = 0
        self.fmin = 1E-8
        self.structures = [None for _ in range(self.Nbins)]
        self.db_name = db_name
        self.current_id = db_id
        self.chem_pot_db_uid = {}

        if (len(self.atoms) != len(self.site_types )):
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

        if ( len(self.possible_swaps) == 0 ):
            self.possible_swaps = []
            for i in range(len(self.site_elements)):
                self.possible_swaps.append({})
                for element in self.site_elements[i]:
                    self.possible_swaps[-1][element] = []
                    for e in self.site_elements[i]:
                        if ( e == element ):
                            continue
                        self.possible_swaps[-1][element].append(e)

    def read_params( self ):
        """
        Reads the entries from a database
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursor()

        cur.execute( "SELECT energy,dos,histogram,fmin,current_f,initial_f,queued from simulations where id=?", self.db_id )
        entries = cur.fetchone()
        cur.execute( "SELECT uid,elements,potential from chemical_potentials where id=?", self.db_id )
        chem_pot_entry = entries.fetchall()
        conn.close()

        queued = entries[6]
        if ( queued != 0 ):
            self.E = convert_array(entries[0])
            self.dos = convert_array(entries[1])
            self.histogram = convert_array(entries[2])
        self.fmin = float( entries[3] )
        self.f = float( entries[4] )
        self.f0 = float( entries[5] )

        # Read the chemical potentials
        for entry in chem_pot_entry:
            self.chem_pot[entry[1]] = float(entry[2])
            self.chem_pot_db_uid[entry[1]] = int(entry[0])

    def get_bin( self, energy ):
        """
        Returns the binning corresponding to the energy
        """
        return int( (energy-self.Emin)*self.Nbins/(self.Emax-self.Emin) )

    def get_energy( self, indx ):
        """
        Returns the energy corresponding to a given bin
        """
        return self.Emin + (self.Emax-self.Emin )*indx/self.Nbins

    def _step( self ):
        """
        Perform one MC step
        """
        indx = np.random.randint(low=0,high=len(self.atoms))
        symb = self.atoms[indx].symbol

        site_type = self.site_types[indx]
        possible_elements = self.possible_swaps[site_type][symb]

        # This is slow and should be optimized
        new_symbol = possible_elements[np.random.randint(low=0,high=len(possible_elements))]
        system_changes = [(indx,new_symbol)]
        self.atoms[indx].symbol = new_symbol
        energy = self.atoms.get_potential_energy()
        #self.calc.calculate( self.atoms, properties=["energy"], system_changes=system_changes )
        chem_pot_change = self.chem_pot[symb]*(self.atoms_count[symb]-1) + self.chem_pot[new_symbol]*(self.atoms_count[new_symbol]+1)
        #energy = self.calc.results["energy"]-chem_pot_change
        energy -= chem_pot_change
        #selected_bin = self.get_bin(energy)

        if ( energy < self.Emin ):
            if ( self.f < self.f0 ):
                return
            self.redistribute_hist(energy,self.Emax)
        elif ( energy >= self.Emax ):
            if ( self.f < self.f0 ):
                return
            self.redistribute_hist(self.Emin,energy)
        selected_bin = self.get_bin(energy)

        rand_num = np.random.rand()
        diff = self.entropy[self.current_bin]-self.entropy[selected_bin]
        if ( diff > 0.0 ):
            accept_ratio = 1.0
        else:
            accept_ratio = np.exp( self.entropy[self.current_bin]-self.entropy[selected_bin] )
        if ( rand_num < accept_ratio  ):
            self.current_bin = selected_bin
            self.atoms_count[symb] -= 1
            self.atoms_count[new_symbol] += 1
            self.atoms[indx].symbol = new_symbol
        else:
            self.atoms[indx].symbol = symb

        if ( self.structures[self.current_bin] is None ):
            # Store the atoms if it has no structure in this bin from before
            self.structures[self.current_bin] = self.atoms.copy()

        self.histogram[self.current_bin] += 1
        self.entropy[self.current_bin] += self.f

    def is_flat( self ):
        """
        Checks if the histogram is flat.
        Note that if the histogram contains less than 100 entries,
        it will return False
        """
        if ( np.sum(self.histogram) < 100 ):
            return False

        mask = np.zeros(len(self.histogram),dtype=np.int8)
        mask[:] = True
        for i in range(len(self.histogram)):
            if ( self.histogram[i] == 0 and self.structures[i] is None ):
                mask[i] = False
        mean = np.mean( self.histogram[mask] )
        #return np.percentile(self.histogram, 10 ) > self.flatness_criteria*mean
        return np.min(self.histogram[mask]) > self.flatness_criteria*mean

    def save( self, fname ):
        """
        Saves the object as a pickle file
        """
        with open(fname,'wb') as ofile:
            pkl.dump(self,ofile)

        base = fname.split(".")[0]
        jsonfname = base+".json"
        self.save_results(jsonfname)

    def save_results( self, fname ):
        """
        Saves the DOS to a JSON file
        """
        data = {}
        data["dos"] = self.dos.astype(float).tolist()
        data["sgc_energy"] = self.E.astype(float).tolist()
        data["f"] = float( self.f )
        data["initial_f"] = float( self.f0 )
        data["flatness_criteria"] = self.flatness_criteria
        data["histogram"] = self.histogram.astype(int).tolist()
        data["fmin"] = float( self.fmin )
        data["converged"] = int( (self.f < self.fmin ) )
        data["chem_pot"] = self.chem_pot

        with open( fname, 'w' ) as outfile:
            str_ = json.dumps(data, indent=4, sort_keys=True, separators=(",",":"), ensure_ascii=False )
            outfile.write(str_)

    def save_db( self ):
        """
        Updates the database with the entries
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        sql = "update simulations set energy="
        cur.execute( "update simulations set energy=?", adapt_array(self.E) )
        cur.execute( "update simulations set dos=?", adapt_array(self.dos) )
        cur.execute( "update simulations set histogram=?", adapt_array(self.histogram))
        conv = int(self.f < self.fmin)
        cur.execute( "update simulations set fmin=?, current_f=?, initial_f=?, converged=? where id=?", (self.fmin,self.f,self.f0,conv,self.db_id) )
        conn.commit()

        # Update the chemical potentials
        for key,value in self.chem_pot:
            cur.execute( "update chemical_potentials set element=?, potential=? where uid=?", (key, value, chem_pot_db_uid[key]) )
        conn.comit()
        conn.close()

    def set_number_of_bins( self, N ):
        """
        Changes the number of bins
        TODO: Does not work yet...
        """
        self.Nbins = N
        self.redistribute_hist(self.Emin,self.Emax)

    def goto_lowest_entropy_structure( self ):
        """
        Change to system to one of the structures that are known to be in the
        low density region
        """
        args = np.argsort(self.histogram)
        for i in range(len(args) ):
            if ( self.structures[args[i]] is None ):
                continue
            self.atoms = self.structures[args[i]]
            self.atoms.set_calculator( self.calc )
            self.current_bin = args[i]
            for key in self.atoms_count.keys():
                self.atoms_count[key]=0
            self.initialize() # Update the atoms count
            return

    def update_range( self ):
        """
        Updates the range
        """
        upper = self.Nbins
        for i in range(len(self.histogram)-1,0,-1):
            if ( self.histogram[i] > 0 ):
                upper = i
                break
        lower = 0
        for i in range(len(self.histogram)):
            if ( self.histogram[i] > 0 ):
                lower = i
                break

        Emin = self.get_energy(lower)
        Emax = self.get_energy(upper)
        if ( Emin != self.Emin or Emax != self.Emax ):
            self.redistribute_hist(Emin,Emax)

    def redistribute_hist( self, Emin, Emax ):
        """
        Redistributes the histograms
        """
        eps = 1E-8
        Emax += eps
        old_energy = self.get_energy(self.current_bin)
        new_E = np.linspace( Emin, Emax, self.Nbins )
        interp_hist = interpolate.interp1d( self.E, self.histogram, bounds_error=False, fill_value=0 )
        new_hist = interp_hist(new_E)
        interp_logdos = interpolate.interp1d( self.E, self.entropy, bounds_error=False, fill_value=0 )
        new_logdos = interp_logdos(new_E)

        # Scale
        if ( np.sum(new_hist) > 0 ):
            new_hist *= np.sum(self.histogram)/np.sum(new_hist)
        if ( np.sum(new_logdos) > 0 ):
            new_logdos *= np.sum(self.entropy)/np.sum(new_logdos)
        old_energies = copy.deepcopy(self.E)
        self.E = new_E
        self.histogram = np.floor(new_hist).astype(np.int32)
        self.entropy = new_logdos
        for i in range(len(self.histogram)):
            if ( self.histogram[i] == 0 ):
                # Set the DOS to 1 if the histogram indicates that it has never been visited
                # This just an artifact of the interpolation and setting it low will make
                # sure that these parts of the space gets explored
                self.entropy[i] = 0.0

        self.Emin = Emin
        self.Emax = Emax
        new_structs = [None for _ in range(self.Nbins)]
        for i in range(len(old_energies)):
            new_indx = self.get_bin(old_energies[i])
            if ( new_indx < 0 or new_indx >= self.Nbins ):
                continue
            new_structs[new_indx] = self.structures[i]
        self.structures = new_structs
        self.current_bin = self.get_bin(old_energy)

    def run( self, maxsteps=10000000 ):
        f_small_enough = False
        low_struct = 200
        for i in range(maxsteps):
            self._step()
            if ( i%self.update_bounds_every == 0 and i > 0 and self.f > 0.5*self.f0 ):
                self.update_range()

            if ( self.is_flat() ):
                self.histogram[:] = 0
                self.f *= 0.5
                #self.entropy -= np.max(self.entropy)
                #self.entropy[self.entropy<-40] = -40 # Reset ti a vanishingly small number

            if ( i%low_struct == 0 and i > 0 ):
                print ("Resetting to a low entropy structure")
                self.goto_lowest_entropy_structure()

            if ( self.f < self.fmin ):
                f_small_enough = True
                break
        self.dos = np.exp(self.entropy - np.mean(self.entropy))
        print ("Current f: {}".format(self.f))

    def sgc_partition_function( self, T ):
        """
        Computes the partition function in the SGC ensemble
        """
        return np.sum( self.dos*self._boltzmann_factor(T) )

    def _boltzmann_factor( self, T ):
        """
        Returns the boltzmann factor
        """
        E0 = np.min(self.E)
        return np.exp( -(self.E-E0)/(units.kB*T) )

    def sgc_avg_energy( self, T ):
        """
        Computes the average energy in the SGC ensemble
        """
        return np.sum( self.E*self.dos*self._boltzmann_factor(T) )/self.sgc_partition_function(T)

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
        return -units.kB*T*( np.log(self.sgc_partition_function(T)) - np.min(self.E)/(units.kB*T) )

    def plot_dos( self ):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( self.E, self.dos, ls="steps" )
        ax.set_yscale("log")
        ax.set_xlabel( "SGC energy (eV)" )
        ax.set_ylabel( "Density of states" )
        return fig

    def plot_histogram( self ):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( self.E, self.histogram, ls="steps" )
        ax.set_xlabel( "SGC energy (eV)" )
        ax.set_ylabel( "Number of times visited")
        return fig

# Helper functions for numpy arrays to SQL database
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out,arr)
    out.seek(0)
    return sq.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)
