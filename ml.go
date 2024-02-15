package main

import (
	"math"
	"math/rand"
)

const (
	randMin, randMax = 1500, 2000
)

func RandWeights() (w []float64, b float64) {
	RandFloat := func() float64 {
		return randMin + rand.Float64()*(randMax-randMin)
	}
	RandArr := func() []float64 {
		arr := make([]float64, argumentCount)
		for i := 0; i < argumentCount; i++ {
			arr[i] = RandFloat()
		}
		return arr
	}
	return RandArr(), RandFloat()
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

func dCost(xs [][]float64, labels, y []float64) (dw []float64, db float64) {
	dw = make([]float64, argumentCount)
	var diff float64
	m := float64(len(xs))
	for i := range xs {
		diff = y[i] - labels[i]
		for j := range dw {
			dw[j] += 1 / m * diff * xs[i][j]
		}
		db += 1 / m * diff
		// Usually do: diff = label - prediction; derivative -= ...
	}
	return dw, db
}

func train(epochCount int, xs [][]float64, ys []float64, lrw, lrb float64, sink func(epoch int, w, dw []float64, b, db float64)) (w []float64, b float64, err error) {
	if len(xs) < 1 {
		// return weights, errors.New("no training examples provided")
		panic("no training examples provided")
	}

	// weights = RandWeights()
	w = make([]float64, 2)
	for epoch := 0; epoch < epochCount; epoch++ {
		dw, db := dCost(xs, ys, inference(xs, w, b))

		// Adjusting weights
		b -= db * lrb
		for i := range w {
			w[i] -= dw[i] * lrw
		}

		sink(epoch, w, dw, b, db)
	}

	return w, b, nil
}

// func decisionBoundaryFunction(x1, x2 float64) func(float64) float64 {

// }
