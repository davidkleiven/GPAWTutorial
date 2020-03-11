package main

import (
	"math"
	"math/rand"

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
func main() {
	N := 256
	eta1 := pf.NewField("eta1", N*N, nil)
	eta2 := pf.NewField("eta2", N*N, nil)
	conc := pf.NewField("conc", N*N, nil)
	uniform(0.2, conc.Data)

	misfit1 := mat.NewDense(3, 3, []float64{0.044, 0.0, 0.0, 0.0, -0.028, 0.0, 0.0, 0.0, 0.044})
	misfit2 := mat.NewDense(3, 3, []float64{-0.028, 0.0, 0.0, 0.0, 0.044, 0.0, 0.0, 0.0, 0.044})
	C_al := []float64{0.62639459, 0.41086487, 0.41086487, 0, 0, 0,
		0.41086487, 0.62639459, 0.41086487, 0, 0, 0,
		0, 0, 0, 0, 0.42750351, 0,
		0, 0, 0, 0, 0, 0.42750351}
	C_al_tensor := elasticity.FromFlatVoigt(C_al)

	dx := 0.5
	dt := 0.01

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
	model.AddEquation("dcont/dt = LAP dfdc + alpha*LAP^2 c")
	model.AddEquation("deta1/dt = dfdn1 + HESS1 + ELAST1")
	model.AddEquation("deta2/dt = dfdn2 + HESS2 + ELAST2")

	// Initialize the solver
	solver := pf.NewSolver(&model, []int{N, N}, dt)
	fileBackup := pf.Float64IO{
		Prefix: "/work/sophus/davidkl/AdaptiveCHGL/ch",
	}
	solver.AddCallback(fileBackup.SaveFields)
	solver.Solve(10, 100)
}
