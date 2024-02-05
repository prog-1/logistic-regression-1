package main

import (
	"math"
)

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-1*z))
}

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
	for i := 0; i < len(inputs); i++ {
		inf = append(inf, dot(inputs[i], w)+b)
	}
	return inf
}
