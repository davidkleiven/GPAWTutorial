import numpy as np
from scipy.stats import skew, kurtosis, ks_2samp
from clease import StructureMapper
from ase.io.trajectory import TrajectoryReader

folder = "data"
trajectories = ["cluster_expansion_almgsiX_fcc.traj",
                "cluster_expansion_fesi_bcc.traj",
                "mgsn.traj", "ce_prebeta_interstitial.traj",
                "LiCrO2F.traj", "LiMnOXpbe1.traj", "LiMnOXpbe2.traj",
                "LiMnOXscan.traj", "LiMnOXhse06.traj"]

#trajectories = ["cluster_expansion_almgsiX_fcc.traj"]

class MapData(object):
    def __init__(self):
        self.chemical_formula = []
        self.success = []
        self.hyd_strain = []
        self.fnorm_deviatoric = []
        self.max_dist = []
        self.mean_dist = []
        self.std_dist = []
        self.skew_dist = []
        self.kurtosis_dist = []
        self.vac_conc = []
        self.min_dist = []
        self.fnorm_corr = []
        self.system = []

    def merge(self, other):
        """
        Merge other into this
        """
        self.success += other.success
        self.hyd_strain += other.hyd_strain
        self.fnorm_deviatoric += other.fnorm_deviatoric
        self.max_dist += other.max_dist
        self.mean_dist += other.mean_dist
        self.std_dist += other.std_dist
        self.skew_dist += other.skew_dist
        self.kurtosis_dist += other.kurtosis_dist
        self.chemical_formula += other.chemical_formula
        self.vac_conc += other.vac_conc
        self.min_dist += other.min_dist
        self.fnorm_corr += other.fnorm_corr
        self.system += other.system

    def save(self, fname):
        with open(fname, 'w') as out:
            out.write("Formula,success,hyd_strain,fnorm_dev_strain,max_dist,"
                      "mean_dist,std_dist,skew_dist,kurtosis_dist,vac_conc,min_dist,"
                      f"form_corr,system\n")
            for i in range(len(self.success)):
                out.write(f"{self.chemical_formula[i]},{int(self.success[i])},"
                          f"{self.hyd_strain[i]},{self.fnorm_deviatoric[i]},"
                          f"{self.max_dist[i]},{self.mean_dist[i]},{self.std_dist[i]},"
                          f"{self.skew_dist[i]},{self.kurtosis_dist[i]},"
                          f"{self.vac_conc[i]},{self.min_dist[i]},{self.fnorm_corr[i]},"
                          f"{self.system[i]}\n")


def revert(images):
    mapper = StructureMapper()
    mapData = MapData()
    weird_images = []
    sucess_disp = []
    failure_disp = []
    for i in range(0, len(images), 2):
        initial = images[i]
        v = initial.get_volume()/len(initial)
        r = (3*v/(4.0*np.pi))**(1.0/3.0)
        final = images[i+1]
        final.wrap()
        recovered, info = mapper.snap_to_lattice(final.copy(), initial.copy())
        info.displacements /= r
        success = np.allclose(recovered.numbers, initial.numbers)
        mapData.success.append(success)
        mapData.max_dist.append(np.max(info.displacements))
        mapData.mean_dist.append(np.mean(info.displacements))
        mapData.std_dist.append(np.std(info.displacements))
        mapData.skew_dist.append(skew(info.displacements))
        mapData.kurtosis_dist.append(kurtosis(info.displacements))
        hyd_strain = np.trace(info.strain)/3.0
        mapData.hyd_strain.append(hyd_strain)
        dev_strain = info.strain - hyd_strain*np.eye(3)
        mapData.fnorm_deviatoric.append(np.linalg.norm(dev_strain, ord='fro'))
        mapData.chemical_formula.append(initial.get_chemical_formula())
        mapData.min_dist.append(np.min(info.displacements))
        numX = sum(1 for atom in recovered if atom.symbol == 'X')
        mapData.vac_conc.append(numX/len(recovered))

        corr = np.corrcoef(info.dispVec.T)
        assert corr.shape == (3, 3)
        #corr -= np.diag(corr)
        tr = np.trace(corr)/3.0
        corr -= tr*np.eye(3)
        fnorm_corr = np.linalg.norm(corr, ord='fro')
        mapData.fnorm_corr.append(fnorm_corr)

        if success:
            sucess_disp += list(info.displacements)
        else:
            failure_disp += list(info.displacements)

        if success and fnorm_corr < 0.3 and np.max(info.displacements) > 1.0:
            print(i)
            weird_images += [initial, recovered, final]

    # from ase.visualize import view
    # try:
    #     view(weird_images)
    # except:
    #     pass
    np.savetxt("data/displacement_dist_success.csv", sucess_disp)
    np.savetxt("data/displacement_dist_failure.csv", failure_disp)
    return mapData

def calculate():
    result = MapData()
    for traj in trajectories:
        images = TrajectoryReader(folder + "/" + traj)
        new_result = revert(images)
        new_result.system = [traj]*len(new_result.success)
        result.merge(new_result)
    result.save("data/map_results.csv")
        
calculate()
