from clease import settingFromJSON
from clease import NewStructures

def main():
    setting = settingFromJSON("almgsixSettings.json")
    setting.db_name = 'almgsiX_dft.db'
    new_struct = NewStructures(setting)
    new_struct.insert_structures(traj_init="data/restricted_gs_5x1x1.traj")

main()
