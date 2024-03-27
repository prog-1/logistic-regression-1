package main

import (
	"errors"
	"fmt"
	"image/color"

	"gonum.org/v1/plot/plotter"
)

// Returns input vector plotters separated by label value(1 or 0)
// Assumes having only 2 classes: 1 and 0
func trainingInputScatters(inputs [][]float64, ys []float64, PosScatterColor, NegScatterColor color.RGBA) (PosScatter, NegScatter *plotter.Scatter, err error) {
	if len(inputs) != len(ys) {
		return PosScatter, NegScatter, errors.New("len(inputs) != len(ys)")
	}

	// Splitting input
	var PosXS, NegXS plotter.XYs
	for i, x := range inputs {
		if ys[i] != 1 && ys[i] != 0 { // Needs to strictly match, so comparison with == is presumably valid
			return PosScatter, NegScatter, fmt.Errorf("ys[%v] ∉ {0,1}", i)
		}
		if ys[i] == 1 {
			PosXS = append(PosXS, plotter.XY{X: x[0], Y: x[1]})
		} else {
			NegXS = append(NegXS, plotter.XY{X: x[0], Y: x[1]})
		}
	}

	// Getting point scatters
	PosScatter, err = plotter.NewScatter(PosXS)
	if err != nil {
		return PosScatter, NegScatter, err
	}
	NegScatter, err = plotter.NewScatter(NegXS)
	if err != nil {
		return PosScatter, NegScatter, err
	}

	PosScatter.Color, NegScatter.Color = PosScatterColor, NegScatterColor

	return PosScatter, NegScatter, nil
}
