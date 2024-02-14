package main

import (
	"math"
)

type WeightRelated struct {
	ws [argumentCount]float64
	b  float64
}

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

func inference(xs [][]float64, w []float64, b float64) (probabilities []float64) {
	for _, x := range xs {
		probabilities = append(probabilities, prediction(x, w, b))
	}
	return probabilities
}

func dCost(xs [][]float64, y, p []float64) (dw []float64, db float64) {
	if len(xs) == 0 {
		return dw, db
	}

	var diff float64
	m := float64(len(xs))
	n := len(xs[0])
	dw = make([]float64, n)
	for i := range xs {
		diff = p[i] - y[i]
		for j := range dw {
			dw[j] += 1 / m * diff * xs[i][j]
		}
		db += 1 / m * diff
	}
	return dw, db
}
