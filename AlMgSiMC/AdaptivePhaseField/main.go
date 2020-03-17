package main

import (
	"flag"
	"math"
	"math/rand"
	"os"
	"runtime/pprof"
	"strings"

	"github.com/davidkleiven/gopf/elasticity"
	"github.com/davidkleiven/gopf/pf"
	"gonum.org/v1/gonum/mat"
)

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
			data[i] = complex(rand.Float64()*0.5, 0.0)
		}
	}
}

func main() {
	dtArg := flag.Float64("dt", 0.001, "Time step in the simulation")
	dxArg := flag.Float64("dx", 0.001, "Spatial discretization used in the simulation")
	initialization := flag.String("init", "uniform", "initialization type. square, uniform or a comma separated list of filenames")
	startEpoch := flag.Int("start", 0, "Epoch to start from")
	numEpoch := flag.Int("epoch", 10, "Number of epochs to run")
	numSteps := flag.Int("steps", 100, "Number of steps per epoch")
	vandevenOrder := flag.Int("vandeven", 0, "Order of the vandeven filter. If 0 no filter will be applied.")
	outfolder := flag.String("folder", "./", "Folder where the output files will be stored")
	cpuprof := flag.String("cpuprof", "", ".prof file where the CPU profile will be stored")
	flag.Parse()

	if *cpuprof != "" {
		f, err := os.Create(*cpuprof)
		if err != nil {
			panic(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	N := 256
	eta1 := pf.NewField("eta1", N*N, nil)
	eta2 := pf.NewField("eta2", N*N, nil)
	conc := pf.NewField("conc", N*N, nil)

	if *initialization == "uniform" {
		uniform(0.2, conc.Data)
	} else if *initialization == "square" {
		square(1.0, conc.Data, N)
		square(0.82, eta1.Data, N)
	} else {
		// Load from file
		fnames := strings.Split(*initialization, ",")
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

	dx := *dxArg
	dt := *dtArg

	// Define gradient coefficients
	alpha_corr := 10.0
	beta11_corr := 2.0
	beta22_corr := 2.0
	beta11 := 8.33/(dx*dx) + beta11_corr
	beta22 := 16.72/(dx*dx) + beta22_corr
	alpha := pf.Scalar{
		Name:  "alpha",
		Value: complex(133.33/(dx*dx)+alpha_corr, 0.0),
	}

	elast1 := pf.NewHomogeneousModolus("eta1", []int{N, N}, C_al_tensor, misfit1)
	elast2 := pf.NewHomogeneousModolus("eta2", []int{N, N}, C_al_tensor, misfit2)

	hess1 := pf.TensorialHessian{
		Field: "eta1",
		K:     []float64{beta11, 0.0, beta22, 0.0},
	}
	hess2 := pf.TensorialHessian{
		Field: "eta2",
		K:     []float64{beta22, 0.0, beta11, 0.0},
	}

	// Initialize the model
	model := pf.NewModel()
	model.AddScalar(alpha)
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
	model.AddEquation("dconc/dt = LAP dfdc - alpha*LAP^2 conc")
	model.AddEquation("deta1/dt = dfdn1 + HESS1 + ELAST1")
	model.AddEquation("deta2/dt = dfdn2 + HESS2 + ELAST2")

	// Initialize the filter

	// Initialize the solver
	solver := pf.NewSolver(&model, []int{N, N}, dt)

	var vandeven pf.Vandeven
	if *vandevenOrder > 0 {
		vandeven = pf.NewVandeven(*vandevenOrder)
		solver.Stepper.SetFilter(&vandeven)
	}
	solver.StartEpoch = *startEpoch
	model.Summarize()
	fileBackup := pf.Float64IO{
		Prefix: *outfolder + "ch",
	}
	solver.AddCallback(fileBackup.SaveFields)
	nepoch := *numEpoch
	solver.Solve(nepoch, *numSteps)
	pf.WriteXDMF(fileBackup.Prefix+".xdmf", []string{"conc", "eta1", "eta2"}, "ch", nepoch+solver.StartEpoch, []int{N, N})
}
