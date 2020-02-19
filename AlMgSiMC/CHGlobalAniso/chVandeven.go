package main

import (
	"flag"
	"fmt"
	"math/rand"

	"github.com/davidkleiven/gopf/pf"
)

func main() {
	folder := flag.String("folder", "./", "folder where output will be stored")
	initFile := flag.String("init", "", "file with the initial concentration distribution")
	flag.Parse()

	nx := 256
	ny := 256
	dt := 0.01
	domainSize := []int{nx, ny}
	model := pf.NewModel()
	conc := pf.NewField("conc", nx*ny, nil)
	model.AddScalar(pf.Scalar{
		Name:  "m1",
		Value: complex(-1.0, 0.0),
	})

	if *initFile == "" {
		// Initialize with random concentration
		r := rand.New(rand.NewSource(0))
		for i := range conc.Data {
			conc.Data[i] = complex(1.0*r.Float64()-1.0, 0.0)
		}
	} else {
		data := pf.LoadFloat64(*initFile)
		for i := range data {
			conc.Data[i] = complex(data[i], 0.0)
		}
		fmt.Print("Loaded data from binary file")
	}

	// Initialize the center
	model.AddField(conc)
	model.AddEquation("dconc/dt = LAP conc^3 + m1*LAP conc")

	// Initialize solver
	solver := pf.NewSolver(&model, domainSize, dt)
	filter := pf.NewVandeven(5)
	solver.Stepper.SetFilter(&filter)

	// Initialize uint8 IO
	out := pf.NewFloat64IO(*folder + "cahnHilliard2D")
	solver.AddCallback(out.SaveFields)

	// Solve the equation
	nepoch := 10
	solver.Solve(nepoch, 1000)
	pf.WriteXDMF(*folder+"cahnHillard.xdmf", []string{"conc"}, "cahnHilliard2D", nepoch, domainSize)
}
