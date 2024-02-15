package main

import (
	"errors"
	"fmt"
	"image/color"
	"log"
	"strconv"

	"gonum.org/v1/plot/plotter"
)

const (
	argumentCount = 2
	classCount    = 2
)

// Reads data from data/exams1.csv and returns [][argumentCount]float64 with labels
func ReadExams1() (xs [][]float64, ys []float64) {
	parseRow := func(row []string) error {
		if len(row) != argumentCount+1 {
			return fmt.Errorf("row length != %v", argumentCount+1)
		}

		var err error
		x := make([]float64, 2)
		for i := range x {
			x[i], err = strconv.ParseFloat(row[i], 64)
			if err != nil {
				return err
			}
		}
		var y float64
		y, err = strconv.ParseFloat(row[argumentCount], 64)
		if err != nil {
			return err
		}

		xs, ys = append(xs, x), append(ys, y)

		return nil
	}
	if err := readCSV("data/exams1.csv", parseRow); err != nil {
		log.Fatal(err)
	}
	return xs, ys
}

// Returns input vector plotters separated by label value(1 or 0)
// Assumes having only 2 classes: 1 and 0
func trainingInputScatters(xs [][argumentCount]float64, ys []float64, scatterColors [2]color.RGBA) (pointScatters [classCount]*plotter.Scatter, err error) {
	if len(xs) != len(ys) {
		return pointScatters, errors.New("len(xs) != len(ys)")
	}

	// Splitting input
	var PosXS, NegXS plotter.XYs
	for i, x := range xs {
		if ys[i] != 1 && ys[i] != 0 { // Needs to strictly match, so comparison with == is presumably valid
			return pointScatters, fmt.Errorf("ys[%v] ∉ {0,1}", i)
		}
		if ys[i] == 1 {
			PosXS = append(PosXS, plotter.XY{X: x[0], Y: x[1]})
		} else {
			NegXS = append(NegXS, plotter.XY{X: x[0], Y: x[1]})
		}
	}

	// Getting point scatters
	pointScatters[0], err = plotter.NewScatter(PosXS)
	if err != nil {
		return pointScatters, err
	}
	pointScatters[1], err = plotter.NewScatter(NegXS)
	if err != nil {
		return pointScatters, err
	}

	// Setting colors
	for i := range pointScatters {
		pointScatters[i].Color = scatterColors[i]
	}

	return pointScatters, nil
}
