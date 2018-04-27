import numpy as np
import json
import os
from scipy import interpolate
import matplotlib as mpl
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
mpl.rcParams["svg.fonttype"] = "none"
from matplotlib import pyplot as plt

fname_template = "data/phase_track/phase_boundary_adaptive_"
outfname = "data/phase_boundary_Al_Al3Mg.json"
N = 100

gau_et_al = {
    "temperature":[353,383],
    "mg_conc":[0.13,0.16]
}
def get_all_results():
    all_data = []
    for i in range(1,N):
        fname = "{}{}.json".format(fname_template,i)
        with open(fname,'r') as infile:
            data = json.load(infile)
        all_data.append(data)
    return all_data

def get_averaged_results( all_data, Tmin, Tmax, nsteps ):
    mu = np.zeros(nsteps)
    comp1 = np.zeros(nsteps)
    comp2 = np.zeros(nsteps)
    mu_sq = np.zeros(nsteps)
    comp1_sq = np.zeros(nsteps)
    comp2_sq = np.zeros(nsteps)

    T = np.linspace( Tmin, Tmax, nsteps )
    counter = np.zeros(nsteps)
    for dset in all_data:
        interp_mu = interpolate.interp1d( dset["temperature"], dset["mu"], bounds_error=False, fill_value=0)
        interp_comp1 = interpolate.interp1d( dset["temperature"], dset["singlet1"], bounds_error=False, fill_value=0 )
        interp_comp2 = interpolate.interp1d( dset["temperature"], dset["singlet2"], bounds_error=False, fill_value=0 )
        maxT = np.max(dset["temperature"])
        new_mu = interp_mu( T )
        new_comp1 = interp_comp1(T)
        new_comp2 = interp_comp2(T)
        mu[T<maxT] += new_mu[T<maxT]
        comp1[T<maxT] += new_comp1[T<maxT]
        comp2[T<maxT] += new_comp2[T<maxT]
        mu_sq[T<maxT] += new_mu[T<maxT]**2
        comp1_sq[T<maxT] += new_comp1[T<maxT]**2
        comp2_sq[T<maxT] += new_comp2[T<maxT]**2
        counter[T<maxT] += 1

    mu /= counter
    comp1 /= counter
    comp2 /= counter

    std_mu = np.sqrt( mu_sq/counter - mu**2 )
    std_comp1 = np.sqrt( comp1_sq/counter - comp1**2 )
    std_comp2 = np.sqrt( comp2_sq/counter - comp2**2 )
    res = {
        "temperature":T.tolist(),
        "mu":mu.tolist(),
        "singlet1":comp1.tolist(),
        "singlet2":comp2.tolist(),
        "std_mu":std_mu.tolist(),
        "std_singlet1":std_comp1.tolist(),
        "std_singlet2":std_comp2.tolist()
    }

    with open(outfname,'w') as outfile:
        json.dump( res, outfile, indent=2, separators=(",",":") )
    print ( "Averaged results written to {}".format(outfname))
    return res

def plot_res( res ):
    fig_mu = plt.figure()
    ax_mu = fig_mu.add_subplot(1,1,1)
    mu = np.array(res["mu"])
    mu -= mu[0]
    mu_minus = mu - np.array( res["std_mu"] )
    mu_pluss = mu + np.array( res["std_mu"] )
    ax_mu.fill_betweenx( res["temperature"], mu_minus*1000.0, mu_pluss*1000.0, color="#d9d9d9" )
    ax_mu.plot( mu*1000.0, res["temperature"] )
    ax_mu.set_xlabel( "\$ \Delta \mu \$ (J/mol)")
    ax_mu.set_ylabel( "Temperature (K)" )
    ax_mu.spines["right"].set_visible(False)
    ax_mu.spines["top"].set_visible(False)

    fig_comp = plt.figure()
    ax_comp = fig_comp.add_subplot(1,1,1)
    comp1 = np.array(res["singlet1"])
    comp2 = np.array( res["singlet2"] )
    mg_conc1 = (1.0-comp1)/2.0
    mg_conc2 = (1.0-comp2)/2.0
    std_mg_conc1 = np.array(res["std_singlet2"])/2.0
    std_mg_conc2 = np.array(res["std_singlet1"])/2.0

    mg_conc1_minus = mg_conc1 - std_mg_conc1
    mg_conc2_minus = mg_conc2 - std_mg_conc2
    mg_conc1_pluss = mg_conc1 + std_mg_conc1
    mg_conc2_pluss = mg_conc2 + std_mg_conc2
    ax_comp.plot( mg_conc1, res["temperature"] )
    ax_comp.plot( mg_conc2, res["temperature"] )
    ax_comp.fill_betweenx( res["temperature"], mg_conc1_minus, mg_conc1_pluss, color="#d9d9d9")
    ax_comp.fill_betweenx( res["temperature"], mg_conc2_minus, mg_conc2_pluss, color="#d9d9d9")
    ax_comp.plot( gau_et_al["mg_conc"], gau_et_al["temperature"], "x" )
    ax_comp.set_xlabel( "Mg concentration" )
    ax_comp.set_ylabel( "Temperature (K)" )
    ax_comp.spines["right"].set_visible(False)
    ax_comp.spines["top"].set_visible(False)

def main():
    all_data = get_all_results()
    res = get_averaged_results( all_data, 100,460,100 )
    plot_res(res)
    plt.show()

if __name__ == "__main__":
    main()
