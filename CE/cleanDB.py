import ase.db

def main():
    db = ase.db.connect( "ceTest.db" )

    # Remove all entries that does not have a gen field

    delID = []
    for row in db.select():
        if ( row.get("gen") is None ):
            delID.append( row.id )
    db.delete(delID)

if __name__ == "__main__":
    main()
