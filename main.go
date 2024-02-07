package main

import (
	"math"
)

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func dot(a []float64, b []float64) float64 {
	var res float64
	if len(a) != len(b) {
		return 0
	}
	for i := 0; i < len(a); i++ {
		res += a[i] * b[i]
	}
	return res
}

func inference(inputs [][]float64, w []float64, b float64) []float64 {
	res := make([]float64, len(inputs))
	for j, x := range inputs {
		f := dot(w, x)
		res[j] = sigmoid(f + b)
	}
	return res
}
