from ase.db import connect

db_name = "data/almgsicu.db"
idFile = 'runIDs.txt'

def single_groups():
    groups = []
    ids = {}
    with connect(db_name) as db:
        con = db.connection
        cur = con.cursor()
        cur.execute("SELECT id, value FROM number_key_values WHERE key='group'")
        for uid, value in cur.fetchall():
            val = int(value)
            groups.append(val)
            ids[val] = uid

    # Remove duplicates
    count = {}
    for g in groups:
        count[g] = count.get(g, 0) + 1

    single = [ids[g] for g, num in count.items() if num < 2]
    return single

def main():
    groups = single_groups()
    
    with open(idFile, 'w') as out:
        for g in groups:
            out.write(f"{g}\n")

main()
