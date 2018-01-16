import ase.db
class SaveToDB(object):
    def __init__(self, db_name, runID, name, mode="both"):
        allowed_modes = ["both","cell","positions"]
        if ( not mode in allowed_modes ):
            raise ValueError( "Mode has to be one of {}".format(allowed_modes) )
        self.db = ase.db.connect( db_name )
        self.runID = runID
        self.name = name
        self.smallestEnergy = 1000.0
        self._is_first = True
        row = self.db.get( id=runID )
        self.fmax = row.get( "fmax", default=None )
        self.smax = row.get( "smax", default=None )
        self.mode = mode

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

            if ( (self.mode == "cell") and (not self.fmax is None) ):
                # Keep the old fmax component if the object relaxes the cell
                self.db.update(id=self.runID,max_force=self.fmax )
            elif ( (self.mode == "positions") and  (not self.smax is None) ):
                # Store the old maximum stress
                self.db.update(id=self.runID,max_stress=self.smax)
