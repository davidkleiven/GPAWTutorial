package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/davidkleiven/gopf/pf"
)

// ModelFunctions is a type used hold parameters needed in the model
type ModelFunctions struct {
	W float64
}

func (m *ModelFunctions) dfdc1(i int, bricks map[string]pf.Brick) complex128 {
	s := real(bricks["mgconc"].Get(i) + bricks["siconc"].Get(i))
	d := real(bricks["mgconc"].Get(i) - bricks["siconc"].Get(i))
	value := 16.0*(2.0*s-6.0*s*s+4.0*s*s*s) + 2.0*m.W*d
	return complex(value, 0.0)
}

func (m *ModelFunctions) dfdc2(i int, bricks map[string]pf.Brick) complex128 {
	s := real(bricks["mgconc"].Get(i) + bricks["siconc"].Get(i))
	d := real(bricks["mgconc"].Get(i) - bricks["siconc"].Get(i))
	value := 16.0*(2.0*s-6.0*s*s+4.0*s*s*s) - 2.0*m.W*d
	return complex(value, 0.0)
}

func main() {
	W := flag.Float64("W", 1.0, "Height of the barrier")
	M := flag.Float64("M", 10.0, "Ratio of the mobility ratios")
	flag.Parse()
	N := 128
	dt := 0.001
	nodes := N * N
	domainSize := []int{N, N}
	model := pf.NewModel()
	mgConc := pf.NewField("mgconc", nodes, nil)
	siConc := pf.NewField("siconc", nodes, nil)

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	avg := 0.2
	// Initialize with random numbers
	for i := range mgConc.Data {
		mgConc.Data[i] = complex(r.Float64()*avg, 0.0)
		siConc.Data[i] = complex(r.Float64()*avg, 0.0)
	}
	model.AddField(mgConc)
	model.AddField(siConc)

	// Define scalars
	mobility := pf.NewScalar("mobility", complex(*M, 0.0))
	k1 := pf.NewScalar("k1", complex(10.0, 0.0))
	minus := pf.NewScalar("minus", complex(-1.0, 0.0))
	model.AddScalar(mobility)
	model.AddScalar(k1)
	model.AddScalar(minus)

	mf := ModelFunctions{
		W: *W,
	}
	// Register functions
	model.RegisterFunction("DFDC1", mf.dfdc1)
	model.RegisterFunction("DFDC2", mf.dfdc2)

	// Define equations
	model.AddEquation("dmgconc/dt = LAP DFDC1 + minus*k1*LAP^2 mgconc")
	model.AddEquation("dsiconc/dt = mobility*LAP DFDC2 + minus*mobility*k1*LAP^2 siconc")

	solver := pf.NewSolver(&model, domainSize, dt)

	prefix := fmt.Sprintf("ch2species_W%d_M%d", int(100.0*mf.W), int(100.0*(*M)))
	folder := "/work/sophus/almgsiMgSiRatio/"
	out := pf.NewFloat64IO(folder + prefix)
	solver.AddCallback(out.SaveFields)

	nEpoch := 1000
	nStepsPerEpoch := 500
	solver.Solve(nEpoch, nStepsPerEpoch)

	pf.WriteXDMF(folder+prefix+".xdmf", []string{"mgconc", "siconc"}, prefix, nEpoch, domainSize)
}
