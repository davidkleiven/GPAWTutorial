from ase.db import connect

def prof():
    db = connect("agpt.db")

    counter = 0
    for row in db.select():
        counter += 1
        calc = row.get('calculator', '')
        name = row.get('name', 'NN')
        gen = row.get('gen', '')
        energy = row.get('energy', '')
        str_type = row.get('struct_type', '')
        cnv = row.get('converged', '')
        size = row.get('size', '')

        print(counter)
    print("Finished")

prof()
