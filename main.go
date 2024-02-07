package main

import (
	"math"
)

func main() {
	// fmt.Printf("z = -1: %f\nz = 0: %f\nz = 1: %f\nz = 2: %f\nz = 3: %f", sigmoid(-1), sigmoid(0), sigmoid(1), sigmoid(2), sigmoid(3))
}

func inference(inputs [][]float64, w []float64, b float64) []float64 {
	var predictions []float64
	for _, x := range inputs {
		predictions = append(predictions, linearModel(x, w, b))
	}
	return predictions
}

func sigmoid(z float64) float64 { return 1 / (1 + math.Pow(math.E, -z)) }

func linearModel(x, w []float64, b float64) (res float64) { return sigmoid(dot(x, w) + b) }

func dot(a []float64, b []float64) (res float64) {
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}
