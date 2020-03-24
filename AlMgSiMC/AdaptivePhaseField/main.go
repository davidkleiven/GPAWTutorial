package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/davidkleiven/gopf/elasticity"
	"github.com/davidkleiven/gopf/pf"
	"gonum.org/v1/gonum/mat"
)

// EtaEq is the equillibrium eta
const EtaEq = 0.81649658092

// Coefficients for 700K
const cSquared = 1.57
const cLin = 0.09
const etaSqConc = 4.16
const etaSq = 3.77
const etaQuad = 8.29
const eta1Sqeta2Quad = 2.76
const beta11Fit = 140.0
const beta22Fit = 878.0

// Coefficients 600K
// const cSquared = 2.944216798279794
// const cLin = 0.17665300789678764
// const etaSqConc = 7.772732347458657
// const etaSq = 7.073186436187378
// const etaQuad = 15.298632023923876
// const eta1Sqeta2Quad = 5.099544007974626
// const beta11Fit = 74.44993782280204
// const beta22Fit = 465.31211139251263

// Coefficients 400K
// const cSquared = 16.643371551928645
// const cLin = 0.9986022931157187
// const etaSqConc = 43.93850089709162
// const etaSq = 39.984035816353376
// const etaQuad = 85.41959789170662
// const eta1Sqeta2Quad = 28.473199297235503
// const beta11Fit = 13.095303260422167
// const beta22Fit = 81.84564537763855

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
	eta1 := real(bricks["eta1"].Get(i)) * EtaEq
	eta2 := real(bricks["eta2"].Get(i)) * EtaEq
	res := 2.0*cSquared*conc - cLin - etaSqConc*(eta1*eta1+eta2*eta2)
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

	eta1 *= EtaEq
	eta2 *= EtaEq
	res := -2*etaSqConc*eta1*conc + 2*etaSq*eta1
	res -= etaQuad * (4.0*math.Pow(eta1, 3) - 2.0*eta1*eta2*eta2 - 6.0*math.Pow(eta1, 5))
	res -= eta1Sqeta2Quad * (2.0*eta1*math.Pow(eta2, 4) + 4.0*math.Pow(eta1, 3)*math.Pow(eta2, 2))
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

func square(value float64, matrix float64, data []complex128, N int, width int) {
	min := N/2 - width/2
	max := N/2 + width/2
	for i := range data {
		ix := i % N
		iy := i / N
		if ix > min && ix < max && iy > min && iy < max {
			data[i] = complex(value, 0.0)
		} else {
			data[i] = complex(matrix, 0.0)
		}
	}
}

// Distribute a set of randomly oriented precipitates
func randomPrecipitates(conc []complex128, eta1 []complex128, eta2 []complex128, width int, num int) {
	N := int(math.Sqrt(float64(len(conc))))
	numScaled := len(conc) / (width * width)

	siteGrid := make([]int, numScaled)
	for i := range siteGrid {
		siteGrid[i] = i
	}

	// Shuffle
	for i := range siteGrid {
		j := rand.Intn(numScaled)
		siteGrid[i], siteGrid[j] = siteGrid[j], siteGrid[i]
	}
	for i := 0; i < num; i++ {
		node := siteGrid[i]
		sx := node % numScaled
		sy := node / numScaled
		x := sx*width + width/2
		y := sy*width + width/2

		orientation := rand.Intn(2)
		var oField []complex128
		if orientation == 0 {
			oField = eta1
		} else {
			oField = eta2
		}

		for i := x - width/2; i < x+width/2; i++ {
			for j := y - width/2; j < y+width/2; j++ {
				if i < 0 {
					i += N
				}
				if j < 0 {
					j += N
				}

				i = i % N
				j = j % N
				idx := i*N + j
				conc[idx] = 1.0
				oField[idx] = 1.0
			}
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

// ConcentrationIndicator returns the concentration
func ConcentrationIndicator(i int, bricks map[string]pf.Brick) complex128 {
	return bricks["conc"].Get(i)
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
	Width    int     `json:"width"`
	NumPrec  int     `json:numPrec`
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
		square(1.0, 0.09/3.14, conc.Data, N, args.Width)
		square(1.0, 0.0, eta1.Data, N, args.Width)
	} else if args.Init == "rndPrec" {
		randomPrecipitates(conc.Data, eta1.Data, eta2.Data, args.Width, args.NumPrec)
	} else {
		// Load from file
		fname := args.Folder + fmt.Sprintf("/ch_conc_%d.bin", args.Start-1)
		concData := pf.LoadFloat64(fname)
		for i := range concData {
			conc.Data[i] = complex(concData[i], 0.0)
		}

		fname = args.Folder + fmt.Sprintf("/ch_eta1_%d.bin", args.Start-1)
		eta1Data := pf.LoadFloat64(fname)
		for i := range eta1Data {
			eta1.Data[i] = complex(eta1Data[i], 0.0)
		}

		fname = args.Folder + fmt.Sprintf("/ch_eta2_%d.bin", args.Start-1)
		eta2Data := pf.LoadFloat64(fname)
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
	beta11 := beta11Fit / (dx * dx)
	beta22 := beta22Fit / (dx * dx)

	M := 1.0 / (dx * dx)
	alpha := pf.Scalar{
		Name:  "alpha",
		Value: complex(0.14*M/(dx*dx), 0.0),
	}

	mobility := pf.Scalar{
		Name:  "mobility",
		Value: complex(-1.0, 0.0),
	}

	elast1 := pf.NewHomogeneousModolus("eta1", []int{N, N}, C_al_tensor, misfit1)
	elast2 := pf.NewHomogeneousModolus("eta2", []int{N, N}, C_al_tensor, misfit2)

	hess1 := pf.TensorialHessian{
		K: []float64{beta11, 0.0, 0.0, beta22},
	}
	hess2 := pf.TensorialHessian{
		K: []float64{beta22, 0.0, 0.0, beta11},
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
	// model.RegisterFunction("ETA1_INDICATOR", smearingDeriv)
	// model.RegisterFunction("CONC_INDICATOR", ConcentrationIndicator)

	// eta1Cons := pf.NewVolumeConservingLP("eta1", "ETA1_INDICATOR", dt, N*N)
	// model.RegisterUserDefinedTerm("ETA1_CONSERVE", &eta1Cons, nil)
	// concCons := pf.NewVolumeConservingLP("conc", "CONC_INDICATOR", dt, N*N)
	// model.RegisterUserDefinedTerm("CONC_CONSERVE", &concCons, nil)
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
	ioutil.WriteFile("data/keyResults.json", b, 0644)
}
