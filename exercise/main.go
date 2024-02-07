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
