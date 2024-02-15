package main

import (
	"errors"
	"fmt"
	"log"
	"strconv"

	"gonum.org/v1/plot/plotter"
)

const (
	argumentCount = 2
	classCount    = 2
)

// TODO: Make it present only in this file, with no dependancies in other files
// Implementes plotter's XYer interface
type Inputs [][argumentCount]float64

func (xs Inputs) Len() int {
	return len(xs)
}

func (xs Inputs) XY(i int) (x, y float64) {
	return xs[i][0], xs[i][1]
}

// Reads data from data/exams1.csv and returns inputs with labels
func readExams1() (xs Inputs, ys []float64) {
	parseRow := func(row []string) error {
		if len(row) != argumentCount+1 {
			return fmt.Errorf("Row length != %v", argumentCount+1)
		}

		var err error
		var x [argumentCount]float64
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

// Separates Inputs with Positive and Negative labels
// Assumes having only 2 classes: 1 and 0
func splitTrainingSet(xs Inputs, ys []float64) (PosXS, NegXS Inputs, err error) {
	if len(xs) != len(ys) {
		return nil, nil, errors.New("len(xs) != len(ys)")
	}

	for i, x := range xs {
		if ys[i] != 1 && ys[i] != 0 { // Needs to strictly match, so comparison with == is presumably valid
			return nil, nil, fmt.Errorf("ys[%v] ∉ {0,1}", i)
		}
		if ys[i] == 1 {
			PosXS = append(PosXS, x)
		} else {
			NegXS = append(NegXS, x)
		}
	}
	return PosXS, NegXS, nil
}

// Returns input vector plotters separated by label value(1 or 0)
// Assumes having only 2 classes: 1 and 0
func trainingInputScatters(PosXS, NegXS Inputs) (pointScatters [classCount]*plotter.Scatter, err error) {
	pointScatters[0], err = plotter.NewScatter(PosXS)
	if err != nil {
		return pointScatters, err
	}
	pointScatters[1], err = plotter.NewScatter(NegXS)
	if err != nil {
		return pointScatters, err
	}
	return pointScatters, nil
}
