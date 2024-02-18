package main

import (
	"math"
)

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Pow(math.E, -z))
}

// func dsigmoid(z float64) float64 {
// 	return (1 / (1 + math.Exp(-1*z))) * (1 - (1 / (1 + math.Exp(-1*z))))
// }

func dot(a []float64, b []float64) float64 {
	if len(a) != len(b) {
		panic("Vectors should have same lenghth!")
	}
	dotProduct := 0.0
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
	}
	return dotProduct

}

func inference(inputs [][]float64, w []float64, b float64) (inf []float64) {
	for _, x := range inputs {
		inf = append(inf, dot(x, w)+b)
	}
	return inf
}

func dCost(inputs [][]float64, y, p []float64) (dw []float64, db float64) {
	dw = make([]float64, len(inputs[0]))

	for i := 0; i < len(inputs); i++ {
		for j := 0; j < len(inputs[0]); j++ {
			dw[j] += (p[i] - y[i]) * inputs[i][j]
		}
	}

	db = 0.0
	for i := 0; i < len(inputs); i++ {
		db += (p[i] - y[i])
	}
	return dw, db
}

func gradientDescent(w []float64, b float64, input [][]float64, y []float64) ([]float64, float64) {
	predictions := inference(input, w, b)
	dw, db := dCost(input, y, predictions)

	for feature := range w {
		w[feature] -= dw[feature] * 0.000001
	}
	b -= db * 5

	return w, b
}
