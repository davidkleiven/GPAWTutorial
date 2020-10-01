package main

import (
	"github.com/MaxHalford/eaopt"
	"github.com/davidkleiven/gogafit/gafit"
)

func main() {
	datafile := "data/cupd.csv"
	data, _ := gafit.Read(datafile, "E_DFT (eV/atom)")
	conf := eaopt.NewDefaultGAConfig()
	conf.PopSize = 30
	ga, _ := conf.NewGA()
	ga.NGenerations = 100

	factory := gafit.LinearModelFactory{
		Config: gafit.LinearModelConfig{
			Data:               data,
			Cost:               gafit.Aicc,
		},
	}

	cost := "Aicc"
	callback := gafit.GABackupCB{
		Cost:       cost,
		Dataset:    data,
		DataFile:   datafile,
		Rate:       10,
		BackupFile: "data/gafit_" + cost + ".json",
	}

	// Add a custom print function to track progress
	ga.Callback = callback.Build()
	ga.Minimize(factory.Generate)
}
