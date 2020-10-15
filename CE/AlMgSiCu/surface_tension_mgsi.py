from ase.io.trajectory import TrajectoryReader
from clease.settings import settings_from_json
from clease.corr_func import CorrFunction

outfile = "data/surfacePure.csv"

def main():
    settings = settings_from_json("data/almgsicu_settings.json")
    cf = CorrFunction(settings)
    traj = TrajectoryReader("../../AlMgSiMC/data/surfacePure.traj")
    corrs = []
    energies = []
    sizes = []
    for i in range(0, len(traj), 2):
        init = traj[i]
        final = traj[i+1]
        corrs.append(cf.get_cf(init))
        sizes.append(len(init))
        energies.append(final.get_potential_energy())

    keys = sorted(corrs[0].keys())
    with open(outfile, 'w') as out:
        header = ",".join(keys)
        header += ",E_DFT,size"
        out.write(header+"\n")
        for c, e, s in zip(corrs, energies, sizes):
            data = ",".join(str(c[k]) for k in keys)
            data += f",{e},{s}\n"
            out.write(data)
    print(f"Data written to {outfile}")
    
main()
