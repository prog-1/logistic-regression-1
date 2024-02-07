package main

import (
	"math"
)

// Clamps the value in range from 0 to 1
func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// Returns a slice vector of inference probabilities for the logistic regression algorithm (p+)
func inference(inputs [][]float64, w []float64, b float64) (res []float64) {

	for x := range inputs {
		res = append(res, sigmoid(dot(inputs[x], w)+b))
	}
	return res
}

// Calculates a dot product of 2 input vectors (a & b are vectors)
func dot(a, b []float64) (res float64) {
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}
