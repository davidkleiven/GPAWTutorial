from ase.ce.settings_bulk import BulkCrystal

conc_args = {
    "conc_ratio_min_1":[[1,0]],
    "conc_ratio_max_1":[[0,1]]
}

bc = BulkCrystal( crystalstructure="fcc", basis_elements=[["Al","Mg"]], db_name="test.db", conc_args=conc_args, size=[4,4,4] )
print (bc.basis_functions)
