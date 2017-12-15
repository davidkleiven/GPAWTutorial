import ase.db
class SaveToDB(object):
    def __init__(self, db_name, runID, name):
        self.db = ase.db.connect( db_name )
        self.runID = runID
        self.name = name
        self.smallestEnergy = 1000.0
        self._is_first = True

    def __call__(self, atoms=None):
        """
        Saves the current run to db if the energy is lower
        """
        if ( atoms is None ):
            return

        if ( self._is_first ):
            self.db.update( self.runID, collapsed=False )
            self._is_first = False

        if ( atoms.get_potential_energy() < self.smallestEnergy ):
            self.smallestEnergy = atoms.get_potential_energy()
            key_value_pairs = self.db.get(name=self.name).key_value_pairs
            del self.db[self.runID]
            self.runID = self.db.write( atoms, key_value_pairs=key_value_pairs )
