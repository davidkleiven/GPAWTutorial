import sqlite3

class WangLandauDBManger( object ):
    def __init__( self, db_name ):

    def check_db( self ):
        """
        Checks if the database has the correct format and updates it if not
        """
        required_fields = {
        "simulations":["id","dos","energy","histogram","fmin","current_f","initial_f","converged","queued"],
        "chemical_potentials":["uid","element","id","potential"]
        }
        required_tables = ["simulations","chemical_potentials"]
        types = {
            "id":"interger",
            "dos":"blob",
            "energy":"blob",
            "histogram":"blob",
            "fmin":"float",
            "current_f":"float",
            "initial_f":"float",
            "converged":"integer",
            "element":"text",
            "potential":"float"
        }

        conn = sq.connect( self.db_name )
        cur = conn.cursor()

        for tabname in required_tables:
            sql = "create table if not exists %s"%(tabname)
            cur.execute(tabname)
        conn.commit()

        # Check if the tables has the required fields
        for tabname in required_tables:
            for col in required_fields[tabname]:
                try:
                    sql = "alter table %s add column %s %s"%(tabname, col, types[col] )
                    cur.execute( sql )
                except:
                    pass
        conn.commit()
        conn.close()

    def get_new_id( self ):
        """
        Get new ID in the simulations table
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursoe()
        cur.execute("SELECT id FROM simulations" ):
        ids = cur.fetchall()
        conn.close()

        if ( len(ids) == 0 ):
            return 0
        return np.max(ids)+1

    def get_new_uid_chem_pot( self ):
        """
        Get new UID in the chemical potential
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "SELECT uid FROM chemical_potentials" )
        uids = cur.fetchall()
        if ( len(uids) == 0 ):
            return 0
        return np.max(uids)+1

    def insert( self, chem_pot, initial_f=2.71, fmin=1E-8, flatness=0.8 ):
        """
        Insert a new entry into the database
        """
        newID = self.get_new_id()
        newUID = self.get_new_uid_chem_pot()

        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "insert into simulations set id=?, initial_f=?, current_f=?, flatness=?, fmin=?, queued=?", (newID, initial_f,initial_f,flatness,fmin,0) )
        cur.commit()

        # Update the chemical potentials
        for key,value in chem_pot.iteritems():
            cur.execute( "insert into chemical_potentials set uid=?, element=?, potential=?, id=?", (newUID,key,value,newID) )
            newUID += 1
        cur.commit()
        conn.close()
