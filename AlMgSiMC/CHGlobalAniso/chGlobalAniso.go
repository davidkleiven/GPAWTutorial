package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"

	"github.com/davidkleiven/gopf/pf"
)

// DerivY allows to a global anisotropy
type DerivY struct {
	Coeff float64
}

// Construct creates the RHS term
func (d *DerivY) Construct(bricks map[string]pf.Brick) pf.Term {
	return func(freq pf.Frequency, t float64, field []complex128) []complex128 {
		brick := bricks["conc"]
		for i := range field {
			f := freq(i)
			prefactor := math.Pow(2.0*math.Pi, 4)
			prefactor *= f[1] * f[1] * (f[0]*f[0] + f[1]*f[1]) * d.Coeff
			field[i] = complex(prefactor, 0.0) * brick.Get(i)
		}
		return field
	}
}

// OnStepFinished does nothing
func (d *DerivY) OnStepFinished(t float64, bricks map[string]pf.Brick) {}

func main() {
	folder := flag.String("folder", "./", "folder where output will be stored")
	aniso := flag.Float64("aniso", 0.0, "global anisotropy")
	initFile := flag.String("init", "", "file with the initial concentration distribution")
	flag.Parse()

	nx := 1024
	ny := 1024
	dt := 0.1
	domainSize := []int{nx, ny}
	model := pf.NewModel()
	conc := pf.NewField("conc", nx*ny, nil)
	gradCoeff := 2.0
	term := DerivY{
		Coeff: gradCoeff * *aniso,
	}

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

	// Add constants
	gamma := pf.NewScalar("gamma", complex(gradCoeff, 0.0)) // Gradient coefficient
	m1 := pf.NewScalar("m1", complex(-1.0, 0.0))            // -1.0
	model.AddScalar(gamma)
	model.AddScalar(m1)
	model.RegisterUserDefinedTerm("DERIVY", &term, nil)

	// Initialize the center
	model.AddField(conc)
	model.AddEquation("dconc/dt = LAP conc^3 + m1*LAP conc + m1*gamma*LAP^2 conc + DERIVY")

	// Initialize solver
	solver := pf.NewSolver(&model, domainSize, dt)

	// Initialize uint8 IO
	out := pf.NewFloat64IO(*folder + "cahnHilliard2D")
	solver.AddCallback(out.SaveFields)

	// Solve the equation
	nepoch := 100
	solver.Solve(nepoch, 1000)
	pf.WriteXDMF(*folder+"cahnHillard.xdmf", []string{"conc"}, "cahnHilliard2D", nepoch, domainSize)
}
