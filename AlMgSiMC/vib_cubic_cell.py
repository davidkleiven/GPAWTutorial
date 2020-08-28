


def prepare(phase):
    db = connect(DB_NAME)
    ref_struct = read(ref_fcc)*(2, 2, 2)

    #db.write(ref_struct, group=1, comment="Reference structure for mgfi FCC")
    structures = generate_mc_rattled_structures(ref_struct, 20, rattle_std,
                                                minimum_distance)
    for s in structures:
        num = random.randint(0, 2**32-1)
        db.write(s, group=num, phase=phase, rattle_std=rattle_std)