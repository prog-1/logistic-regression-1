package main

import (
	"math"
)

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Pow(math.E, -z))
}

func dot(a []float64, b []float64) (res float64) {
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}

func prediction(x []float64, w []float64, b float64) float64 {
	return sigmoid(dot(w, x) + b)
}

func inference(inputs [][]float64, w []float64, b float64) (probabilities []float64) {
	for _, x := range inputs {
		probabilities = append(probabilities, prediction(x, w, b))
	}
	return probabilities
}
