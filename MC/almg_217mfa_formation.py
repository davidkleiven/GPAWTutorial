import json
from matplotlib import pyplot as plt
import glob
import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from ase.units import kJ,mol
from matplotlib import cm

save_free_eng = True
free_eng_file = "data/almg_217mfs_free_eng.json"
def main():
    T = np.array([10000,9000,8000,7000,6000,5000,4000,3000,2000,1000,800,700,600,500,400,375,350,325,300,275,250,225,200,175,150])
    mg_concs = []
    enthalpies = []
    free_energy = []
    ref_energies = {
        "Al":-3.73667187,
        "Mg":-1.59090625
    }
    temps = []
    concs = []
    filenames = glob.glob("data/mfa217/*")
    ignore_indx = []
    for i,filename in enumerate(filenames):
        with open(filename,'r') as infile:
            data = json.load(infile)
        mg_concs.append( data["conc"]["Mg"] )
        concs.append(data["conc"] )
        print (data["conc"],filename)
        enthalpies.append( np.array(data["internal_energy"])/data["natoms"] - ref_energies["Mg"]*data["conc"]["Mg"] - ref_energies["Al"]*data["conc"]["Al"] )
        free_energy.append( np.array(data["free_energy"])/data["natoms"] - ref_energies["Mg"]*data["conc"]["Mg"] - ref_energies["Al"]*data["conc"]["Al"] )
        try:
            temps.append( data["temperature"] )
        except Exception as exc:
            ignore_indx.append(i)
            print("Error occured in {}. Message: {}".format(filename,str(exc)))

    if ( save_free_eng ):
        res = {}
        i = 0
        for indx in range(len(filenames)):
            if ( indx in ignore_indx ):
                continue
            res[filenames[i]] = {}
            res[filenames[i]]["conc"] = concs[i]
            res[filenames[i]]["free_energy"] = free_energy[i].tolist()
            res[filenames[i]]["temperature"] = temps[i]
            i += 1

        with open(free_eng_file,'w') as outfile:
            json.dump( res, outfile, indent=2, separators=(",",":") )
        print ("Free energies written to {}".format(free_eng_file))
    figH = plt.figure()
    axH = figH.add_subplot(1,1,1)
    figF = plt.figure()
    axF = figF.add_subplot(1,1,1)
    figS = plt.figure()
    axS = figS.add_subplot(1,1,1)
    Tmax = 800
    Tmin = 150
    print (len(temps[0]))
    print (len(enthalpies[0]))
    for i in range(len(enthalpies[0])):
        try:
            T = temps[0][i]
            if ( T > Tmax ):
                continue
            srt_indx = np.argsort(mg_concs)
            H = [enthalpies[j][i] for j in range(len(enthalpies))]
            F = [free_energy[j][i] for j in range(len(free_energy))]

            srt_mg = [mg_concs[indx] for indx in srt_indx]
            H = np.array( [H[indx] for indx in srt_indx] )
            F = np.array( [F[indx] for indx in srt_indx] )
            mapped_T = float(T-Tmin)/(Tmax-Tmin)
            S = (H-F)/T
            axH.plot( srt_mg, H*mol/kJ, marker="x", color=cm.copper(mapped_T) )
            axF.plot( srt_mg, F*mol/kJ, marker="o", color=cm.copper(mapped_T) )
            axS.plot( srt_mg, 1E6*S*mol/kJ, marker="o", color=cm.copper(mapped_T))
        except Exception as exc:
            print(str(exc))

    axH.set_xlabel( "Mg concentration" )
    axH.set_ylabel( "Enthalpy of Formation (kJ/mol)")
    axF.set_xlabel( "Mg concentration" )
    axF.set_ylabel( "Free Energy of Formation (kJ/mol)" )
    axS.set_xlabel( "Mg concentration" )
    axS.set_ylabel( "Entropy (mJ/K mol)")
    axH.spines["right"].set_visible(False)
    axH.spines["top"].set_visible(False)
    axF.spines["right"].set_visible(False)
    axF.spines["top"].set_visible(False)
    axS.spines["right"].set_visible(False)
    axS.spines["top"].set_visible(False)

    plt.show()

if __name__ == "__main__":
    main()
