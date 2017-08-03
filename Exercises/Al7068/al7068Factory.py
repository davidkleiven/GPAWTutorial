from ase.lattice.cubic import SimpleCubicFactory
import numpy as np
import copy
import itertools as itools
import random as rnd

class Al7068Factory( SimpleCubicFactory ):
    bravais_basis = []
    element_basis = []
    def __init__( self, atom_percentage={"Al":0.768,
                                        "Zn":0.166,
                                        "Mg":0.0198,
                                        "Cu":0.02,
                                        "Zr":0.0036},
                n_sites_in_each_direction=5 ):
        bravais_basis = []
        element_basis = []
        self.atom_percentage = atom_percentage
        self.n_sites_in_each_direction = n_sites_in_each_direction

        max_bravais = n_sites_in_each_direction
        x_bravais = np.linspace( 0.0, max_bravais, n_sites_in_each_direction )

        x_interstitial = x_bravais + 0.25

        Al7068Factory.bravais_basis = list( itools.product( x_bravais, repeat=3 ) )
        Al7068Factory.bravais_basis += list( itools.product( x_interstitial, repeat=3 ) )

        self._placeSubstitional()
        self._placeInterstitial()

        print ( len(Al7068Factory.bravais_basis), len(Al7068Factory.element_basis))

        # Delete vacancies
        current = 0
        for i in range( len(Al7068Factory.element_basis) ):
            if ( Al7068Factory.element_basis[current] == 5 ):
                del Al7068Factory.bravais_basis[current]
                del Al7068Factory.element_basis[current]
                current -= 1
            current += 1
        Al7068Factory.element_basis = tuple( Al7068Factory.element_basis )

    def _placeSubstitional( self ):
        """
        Places the substitional atoms. Here this is Aluminum and Magnesium.
        """
        n_al_atoms = int( self.atom_percentage["Al"]*self.n_sites_in_each_direction**3 )
        n_mg_atoms = int( self.atom_percentage["Mg"]*self.n_sites_in_each_direction**3 )
        temp_elem_basis = np.zeros(self.n_sites_in_each_direction**3, dtype=np.uint8 )
        temp_elem_basis[:n_al_atoms] = 0
        temp_elem_basis[n_al_atoms:] = 1
        rnd.shuffle( temp_elem_basis )
        Al7068Factory.element_basis = list( temp_elem_basis )

    def _placeInterstitial( self ):
        """
        Places the interstital atoms. Here these are Zn, Cu, Zr
        """
        n_zn_atoms = int( self.atom_percentage["Zn"]*self.n_sites_in_each_direction**3 )
        n_cu_atoms = int( self.atom_percentage["Cu"]*self.n_sites_in_each_direction**3 )
        n_zr_atoms = int( self.atom_percentage["Zr"]*self.n_sites_in_each_direction**3 )

        temp_elem_basis = np.zeros( self.n_sites_in_each_direction**3, dtype=np.uint8 )
        temp_elem_basis[:n_zn_atoms] = 2
        temp_elem_basis[n_zn_atoms:n_zn_atoms+n_cu_atoms] = 3
        temp_elem_basis[ n_zn_atoms+n_cu_atoms:n_zn_atoms+n_cu_atoms+n_zr_atoms] = 4
        temp_elem_basis[ n_zn_atoms+n_cu_atoms+n_zr_atoms: ] = 5 # Vacancy
        rnd.shuffle( temp_elem_basis )
        Al7068Factory.element_basis += list( temp_elem_basis )

Al7068 = Al7068Factory()
