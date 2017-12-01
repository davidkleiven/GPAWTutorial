import sqlite3 as sq
import numpy as np
from wang_landau_scg import WangLandauSGC
from wl_analyzer import WangLandauSGCAnalyzer
import wltools

class WangLandauDBManger( object ):
    def __init__( self, db_name ):
        self.db_name = db_name
        self.check_db()

    def check_db( self ):
        """
        Checks if the database has the correct format and updates it if not
        """
        required_fields = {
        "simulations":["uid","dos","energy","histogram","fmin","current_f","initial_f","converged",
        "queued","Nbins","flatness","groupID","growth_variance","Emin","Emax","initialized","struct_file",
        "gs_energy"],
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
            "potential":"float",
            "Nbins":"integer",
            "flatness":"float",
            "queued":"integer",
            "groupID":"integer",
            "growth_variance":"blob",
            "Emin":"float",
            "Emax":"float",
            "initialized":"integer",
            "struct_file":"text",
            "gs_energy":"float"
        }

        conn = sq.connect( self.db_name )
        cur = conn.cursor()

        for tabname in required_tables:
            sql = "create table if not exists %s (uid integer)"%(tabname)
            cur.execute(sql)
        conn.commit()

        # Check if the tables has the required fields
        for tabname in required_tables:
            for col in required_fields[tabname]:
                try:
                    sql = "alter table %s add column %s %s"%(tabname, col, types[col] )
                    cur.execute( sql )
                except Exception as exc:
                    pass
        conn.commit()
        conn.close()

    def get_new_id( self ):
        """
        Get new ID in the simulations table
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute("SELECT uid FROM simulations" )
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

    def get_new_group( self ):
        """
        Gets new group ID
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "SELECT groupID FROM simulations" )
        groups = cur.fetchall()
        conn.close()
        only_groups = [entry[0] for entry in groups]
        if ( len(only_groups) == 0 ):
            return 0
        return max(only_groups)+1

    def insert( self, chem_pot, initial_f=2.71, fmin=1E-8, flatness=0.8, Nbins=50, energy_range="auto" ):
        """
        Insert a new entry into the database
        """
        newID = self.get_new_id()
        newUID = self.get_new_uid_chem_pot()

        group = self.get_new_group()
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "insert into simulations (uid,initial_f,current_f,flatness,fmin,queued,Nbins,groupID) values (?,?,?,?,?,?,?,?)", (newID, initial_f,initial_f,flatness,fmin,0,Nbins,group) )
        cur.execute( "update simulations set initialized=? where uid=?", (0,newID) )
        conn.commit()

        if ( isinstance(energy_range,list) or isinstance(energy_range,tuple) ):
            if ( len(energy_range) == 2 ):
                Emin = energy_range[0]
                Emax = energy_range[1]
                cur.execute( "update simulations set Emin=?, Emax=? where uid=?",(Emin,Emax,newID))
                E = np.linspace( Emin,Emax+1E-8,Nbins )
                cur.execute( "update simulations set energy=? where uid=?", (wltools.convert_array(E),newID) )
                cur.execute( "update simulations set initialized=? where uid=?", (1,newID) )
                conn.commit()

        # Update the chemical potentials
        for key,value in chem_pot.iteritems():
            cur.execute( "insert into chemical_potentials (uid,element,potential, id) values (?,?,?,?)", (newUID,key,value,newID) )
            newUID += 1
        conn.commit()
        conn.close()

    def add_run_to_group( self, groupID, n_entries=1 ):
        """
        Adds a run to a group
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "SELECT uid,initial_f,current_f,flatness,fmin,queued,Nbins,groupID,Emin,Emax,initialized,energy,gs_energy FROM simulations WHERE groupID=?", (groupID,))
        entries = cur.fetchone()
        oldID = int(entries[0])
        newID = self.get_new_id()
        entries = list(entries)
        cur.execute( "SELECT element,potential FROM chemical_potentials WHERE id=?", (oldID,) )
        chem_pot_entries = cur.fetchall()

        for _ in range(n_entries):
            entries[0] = newID
            entries[7] = groupID
            entries[5] = 0 # Set queued flag to 0
            entries = tuple(entries)
            cur.execute( "INSERT INTO simulations (uid,initial_f,current_f,flatness,fmin,queued,Nbins,groupID,Emin,Emax,initialized,energy,gs_energy) values (?,?,?,?,?,?,?,?,?,?,?,?,?)", entries )
            newUID = self.get_new_uid_chem_pot()
            for entry in chem_pot_entries:
                cur.execute( "INSERT INTO chemical_potentials (uid,element,potential,id) VALUES (?,?,?,?)", (newUID,entry[0],entry[1],newID) )
                newUID += 1
            entries = list(entries)
            newID += 1
        conn.commit()
        conn.close()

    def get_converged_wl_objects( self, atoms, calc ):
        """
        Get a list of all converged Wang-Landau simulations
        """
        conn = sq.connect( self.db_name )
        cur.execute( "SELECT UID FROM simulations WHERE converged=1" )
        uids = cur.fetchall()
        conn.close()
        return self.get_wl_objects( atoms, calc, uids )

    def get_wl_objects( self, atoms, calc, uids ):
        """
        Returns a list of Wang Landau objects corresponding to ids
        """
        obj = []
        for uid in uids:
            objs.append( WangLandauSGC( atoms, calc, self.db_name, uid ) )

    def get_analyzer( self, groupID, min_number_of_converged=1 ):
        """
        Returns a Wang-Landau Analyzer object based on the average of all converged runs
        within a groupID
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "SELECT energy,dos,uid,gs_energy FROM simulations WHERE converged=1 AND groupID=?", (groupID,) )
        entries = cur.fetchall()
        conn.close()
        if ( len(entries) < min_number_of_converged ):
            return None
        uid = int( entries[0][2] )
        energy = wltools.convert_array( entries[0][0] )
        logdos = np.log( wltools.convert_array( entries[0][1] ) )

        # Normalize before averaging
        all_sum = np.sum(np.exp(logdos))
        logdos -= np.log(all_sum)
        gs_energy = np.min( [entries[i][3] for i in range(0,len(entries))] )
        for i in range(1,len(entries)):
            logdos += np.log( wltools.convert_array( entries[i][1] ) )

        logdos /= len(entries)
        dos = np.exp(logdos)

        # Extract chemical potentials
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "SELECT element,potential FROM chemical_potentials WHERE id=?", (uid,) )
        entries = cur.fetchall()

        conn.close()
        chem_pot = {}
        for entry in entries:
            chem_pot[entry[0]] = entry[1]
        return WangLandauSGCAnalyzer( energy, dos, chem_pot, gs_energy )

    def get_analyzer_all_groups( self, min_number_of_converged=1 ):
        """
        Returns a list of analyzer objects
        """
        maxgroup = self.get_new_group()
        analyzers = [self.get_analyzer(i,min_number_of_converged=min_number_of_converged) for i in range(maxgroup)]
        filtered = [entry for entry in analyzers if not entry is None] # Remove non converged entries
        return filtered
