package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/davidkleiven/gopf/elasticity"
	"github.com/davidkleiven/gopf/pf"
	"gonum.org/v1/gonum/mat"
)

// SoluteConcentrationMonitor trackts the average concentration in the matrix
type SoluteConcentrationMonitor struct {
	Data      []float64
	Name      string
	Threshold float64
	NumPoints int
}

// Add adds a new item to the Data
func (scm *SoluteConcentrationMonitor) Add(bricks map[string]pf.Brick) {
	eta1 := bricks["eta1"]
	eta2 := bricks["eta2"]
	conc := bricks["conc"]

	avg := 0.0
	count := 0
	for i := 0; i < scm.NumPoints; i++ {
		if real(eta1.Get(i)) < scm.Threshold && real(eta2.Get(i)) < scm.Threshold {
			count++
			avg += real(conc.Get(i))
		}
	}
	avg /= float64(count)
	scm.Data = append(scm.Data, avg)
	fmt.Printf("Average solute concentration %f\n", avg)
}

func dfdc(i int, bricks map[string]pf.Brick) complex128 {
	conc := real(bricks["conc"].Get(i))
	eta1 := real(bricks["eta1"].Get(i))
	eta2 := real(bricks["eta2"].Get(i))
	res := 2.0*1.57*conc - 0.09 - 4.16*(eta1*eta1+eta2*eta2)
	return complex(res, 0.0)
}

func dfdeta(i int, bricks map[string]pf.Brick, value int) float64 {
	conc := real(bricks["conc"].Get(i))
	var eta1 float64
	var eta2 float64
	if value == 1 {
		eta1 = real(bricks["eta1"].Get(i))
		eta2 = real(bricks["eta2"].Get(i))
	} else {
		// Switch eta1 and eta2 since the expression is symmetrice with respect to thos
		eta1 = real(bricks["eta2"].Get(i))
		eta2 = real(bricks["eta1"].Get(i))
	}

	res := -2*4.16*eta1*conc + 2*3.77*eta1
	res -= 8.29 * (4.0*math.Pow(eta1, 3) - 2.0*eta1*eta2*eta2 - 6.0*math.Pow(eta1, 5))
	res -= 2.76 * (2.0*eta1*math.Pow(eta2, 4) + 4.0*math.Pow(eta1, 3)*math.Pow(eta2, 2))
	return res
}

func dfdn1(i int, bricks map[string]pf.Brick) complex128 {
	return complex(-dfdeta(i, bricks, 1), 0.0)
}

func dfdn2(i int, bricks map[string]pf.Brick) complex128 {
	return complex(-dfdeta(i, bricks, 2), 0.0)
}

func uniform(maxval float64, data []complex128) {
	for i := range data {
		data[i] = complex(rand.Float64()*maxval, 0.0)
	}
}

func square(value float64, data []complex128, N int) {
	min := 7 * N / 16
	max := 9 * N / 16
	for i := range data {
		ix := i % N
		iy := i / N
		if ix > min && ix < max && iy > min && iy < max {
			data[i] = complex(1.0, 0.0)
		} else {
			data[i] = complex(rand.Float64()*0.1, 0.0)
		}
	}
}

func smearingDeriv(i int, bricks map[string]pf.Brick) complex128 {
	x := real(bricks["eta1"].Get(i))
	if x < 0.0 || x > 1.0 {
		return 0.0
	}
	return complex(6.0*x-6.0*x*x, 0.0)
}

// Arguments is a structure that holds all the input arguments
type Arguments struct {
	Dx       float64 `json:"dx"`
	Vandeven int     `json:"vandeven"`
	Dt       float64 `json:"dt"`
	Epoch    int     `json:"epoch"`
	Steps    int     `json:"steps"`
	Folder   string  `json:"folder"`
	Start    int     `json:"start"`
	Init     string  `json:"init"`
}

// KeyResults is a dictionary that stores key numbers from the run
type KeyResults struct {
	FinalSoluteConc float64
}

func main() {
	inputfile := os.Args[1]
	jsonFile, err := os.Open(inputfile)
	if err != nil {
		panic(err)
	}
	byteValue, _ := ioutil.ReadAll(jsonFile)
	var args Arguments
	json.Unmarshal(byteValue, &args)

	seed := time.Now().UnixNano()
	rand.Seed(seed)

	N := 256
	eta1 := pf.NewField("eta1", N*N, nil)
	eta2 := pf.NewField("eta2", N*N, nil)
	conc := pf.NewField("conc", N*N, nil)

	if args.Init == "uniform" {
		uniform(0.1, conc.Data)
	} else if args.Init == "square" {
		square(1.0, conc.Data, N)
		square(0.82, eta1.Data, N)
	} else {
		// Load from file
		fnames := strings.Split(args.Init, ",")
		concData := pf.LoadFloat64(fnames[0])
		for i := range concData {
			conc.Data[i] = complex(concData[i], 0.0)
		}

		eta1Data := pf.LoadFloat64(fnames[1])
		for i := range eta1Data {
			eta1.Data[i] = complex(eta1Data[i], 0.0)
		}

		eta2Data := pf.LoadFloat64(fnames[2])
		for i := range eta2Data {
			eta2.Data[i] = complex(eta2Data[i], 0.0)
		}
	}

	misfit1 := mat.NewDense(3, 3, []float64{0.044, 0.0, 0.0, 0.0, -0.028, 0.0, 0.0, 0.0, 0.044})
	misfit2 := mat.NewDense(3, 3, []float64{-0.028, 0.0, 0.0, 0.0, 0.044, 0.0, 0.0, 0.0, 0.044})
	C_al := []float64{0.62639459, 0.41086487, 0.41086487, 0, 0, 0,
		0.41086487, 0.62639459, 0.41086487, 0, 0, 0,
		0, 0, 0, 0, 0.42750351, 0,
		0, 0, 0, 0, 0, 0.42750351}
	C_al_tensor := elasticity.FromFlatVoigt(C_al)

	dx := args.Dx
	dt := args.Dt

	// Define gradient coefficients
	// beta11 := 9.15 / (dx * dx)
	// beta22 := 16.83 / (dx * dx)
	beta11 := 140.0 / (dx * dx)
	beta22 := 878.0 / (dx * dx)

	M := 1.0 / (dx * dx)
	alpha := pf.Scalar{
		Name:  "alpha",
		Value: complex(0.14*M/(dx*dx), 0.0),
	}

	mobility := pf.Scalar{
		Name:  "mobility",
		Value: complex(M, 0.0),
	}

	elast1 := pf.NewHomogeneousModolus("eta1", []int{N, N}, C_al_tensor, misfit1)
	elast2 := pf.NewHomogeneousModolus("eta2", []int{N, N}, C_al_tensor, misfit2)

	hess1 := pf.TensorialHessian{
		K: []float64{beta11, 0.0, 0.0, beta22},
	}
	hess2 := pf.TensorialHessian{
		K: []float64{beta22, 0.0, beta11},
	}

	// Initialize the model
	model := pf.NewModel()
	model.AddScalar(alpha)
	model.AddScalar(mobility)
	model.AddField(conc)
	model.AddField(eta1)
	model.AddField(eta2)

	model.RegisterUserDefinedTerm("ELAST1", elast1, nil)
	model.RegisterUserDefinedTerm("ELAST2", elast2, nil)
	model.RegisterUserDefinedTerm("HESS1", &hess1, nil)
	model.RegisterUserDefinedTerm("HESS2", &hess2, nil)
	model.RegisterFunction("dfdc", dfdc)
	model.RegisterFunction("dfdn1", dfdn1)
	model.RegisterFunction("dfdn2", dfdn2)
	model.RegisterFunction("ETA1_INDICATOR", smearingDeriv)
	//eta1Cons := pf.NewVolumeConservingLP("eta1", "ETA1_INDICATOR", dt, N*N)
	//model.RegisterUserDefinedTerm("ETA1_CONSERVE", &eta1Cons, nil)
	//model.RegisterUserDefinedTerm("CONS_NOISE", &cnsvNoise, dfields)
	// kT := 0.086 * 200
	// fDeriv := 2.0 * 3.77
	// noise := pf.WhiteNoise{
	// 	Strength: 0.5 * kT / (math.Sqrt(dt) * fDeriv),
	// }
	// model.RegisterFunction("WHITE_NOISE", noise.Generate)

	// specVisc := pf.SpectralViscosity{
	// 	Power:                2,
	// 	DissipationThreshold: 0.25,
	// 	Eps:                  5.0,
	// }
	// model.RegisterUserDefinedTerm("SPECTRAL_VISC", &specVisc, nil)
	model.AddEquation("dconc/dt = mobility*LAP dfdc")
	model.AddEquation("deta1/dt = dfdn1 + HESS1*eta1 + ELAST1")
	model.AddEquation("deta2/dt = dfdn2 + HESS2*eta2 + ELAST2")

	avgConc := SoluteConcentrationMonitor{
		Data:      []float64{},
		Name:      "SoluteConcMonitor",
		Threshold: 0.1,
		NumPoints: len(conc.Data),
	}
	// Initialize the solver
	solver := pf.NewSolver(&model, []int{N, N}, dt)
	solver.AddMonitor(&avgConc)

	var vandeven pf.Vandeven
	if args.Vandeven > 0 {
		vandeven = pf.NewVandeven(args.Vandeven)
		solver.Stepper.SetFilter(&vandeven)
	}
	solver.StartEpoch = args.Start
	model.Summarize()
	fileBackup := pf.Float64IO{
		Prefix: args.Folder + "ch",
	}
	solver.AddCallback(fileBackup.SaveFields)
	nepoch := args.Epoch
	solver.Solve(nepoch, args.Steps)
	pf.WriteXDMF(fileBackup.Prefix+".xdmf", []string{"conc", "eta1", "eta2"}, "ch", nepoch+solver.StartEpoch, []int{N, N})

	keyResults := KeyResults{
		FinalSoluteConc: avgConc.Data[len(avgConc.Data)-1],
	}

	b, err := json.Marshal(keyResults)
	ioutil.WriteFile("keyResults.json", b, 0644)
}
