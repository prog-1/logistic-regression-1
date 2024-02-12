package main

import (
	"fmt"
	"math"
)

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
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

func dCost(inputs [][]float64, y, p []float64) (dw []float64, db float64) {
	if len(inputs) == 0 {
		return dw, db
	}

	var diff float64
	m := float64(len(inputs))
	n := len(inputs[0])
	dw = make([]float64, n)
	for i := range inputs {
		diff = p[i] - y[i]
		for j := range dw {
			dw[j] += 1 / m * diff * inputs[i][j]
		}
		db += 1 / m * diff
	}
	return dw, db
}

func main() {
	students := readStudentsFromCSV()
	for _, student := range students {
		fmt.Println(student)
	}
}
