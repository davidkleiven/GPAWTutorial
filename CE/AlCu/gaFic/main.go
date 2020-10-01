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
	ga.NGenerations = 1000

	fic := gafit.PredictionErrorFIC{
		Data: []int{2, 69, 8, 137, 12},
	}

	factory := gafit.LinearModelFactory{
		Config: gafit.LinearModelConfig{
			Data: data,
			//Cost: gafit.Aicc,
			Cost:         fic.Evaluate,
			MutationRate: 0.2,
		},
	}

	cost := "fic"
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

	// Calculate prediction errors
	model, _ := gafit.ReadModel(callback.BackupFile)
	pred := gafit.GetPredictions(data, model)
	gafit.SavePredictions("data/predictions_"+cost+".csv", pred)
}
