package main

import (
	"math"
	"math/rand"
)

const (
	randMin, randMax = 1500, 2000
)

// TODO: Remove this
type WeightRelated struct {
	ws [argumentCount]float64
	b  float64
}

func (weights *WeightRelated) adjustWeights(derivatives *WeightRelated, learningRate float64) {
	weights.b -= derivatives.b * 0.5
	for i := range weights.ws {
		weights.ws[i] -= derivatives.ws[i] * learningRate
	}
}

func RandWeights() WeightRelated {
	RandFloat := func() float64 {
		return randMin + rand.Float64()*(randMax-randMin)
	}
	RandArr := func() (arr [argumentCount]float64) {
		for i := 0; i < argumentCount; i++ {
			arr[i] = RandFloat()
		}
		return arr
	}
	return WeightRelated{ws: RandArr(), b: RandFloat()}
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

func inference(xs Inputs, w []float64, b float64) (probabilities []float64) {
	for _, x := range xs {
		probabilities = append(probabilities, prediction(x[:], w, b))
	}
	return probabilities
}

func dCost(xs Inputs, labels, y []float64) (dw, db []float64) {
	var derivatives WeightRelated
	var diff float64
	m := float64(len(xs))
	for i := range xs {
		diff = y[i] - labels[i]
		for j := range derivatives.ws {
			derivatives.ws[j] += 1 / m * diff * xs[i][j]
		}
		derivatives.b += 1 / m * diff
	}
	return derivatives
}

// func decisionBoundaryFunction(x1, x2 float64) func(float64) float64 {

// }
