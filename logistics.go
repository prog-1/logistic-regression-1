package main

import (
	"math"
)

const (
	lr     = 0.001
	epochs = 1000
)

// -------------------- Part I --------------------

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

func prediction(x []float64, w []float64, b float64) (res float64) {
	return sigmoid(dot(x, w) + b)
}

// Calculates a dot product of 2 input vectors (a & b are vectors)
func dot(a, b []float64) (res float64) {
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}

// -------------------- Part II --------------------

// // Calculates the cost function gradient
// func dCost(inputs [][]float64, y, p []float64) ([]float64, float64) {
// 	m := float64(len(inputs))          // number of input vectors x
// 	totalE := 0.0                      // sum of the difference between prediction and truth (errors) for all points
// 	dw := make([]float64, len(inputs)) // slice of weight gradients

// 	for i, x := range inputs { //for every point
// 		e := p[i] - y[i] //difference between i'th prediction and truth
// 		totalE += e      //add current difference into piggy bank

// 		for feature := range x { //for every feature
// 			dw[feature] += e * x[feature] //update weight gradient for this feature of this point
// 		}
// 	}

// 	for i := range dw { //for every weight gradient
// 		dw[i] /= m // get average weight gradient for every feature
// 	}

// 	db := totalE / m // get average bias gradient

// 	return dw, db
// }

// Calculates the cost function gradient
func dCost(inputs [][]float64, y, p []float64) ([]float64, float64) {
	db := 0.0
	dw := make([]float64, len(inputs[0]))

	m := float64(len(inputs)) //number of points

	//inferences := inference(inputs, dw, db)

	for i, x := range inputs { // for every point
		prediction := prediction(x, dw, db)
		for j, feature := range x {
			dw[j] += (prediction - y[i]) * feature / m
		}
		db += (prediction - y[i]) / m
	}

	return dw, db
}

func gradientDescent(dw []float64, db float64) ([]float64, float64) {
	var b float64
	b -= db * lr

	w := make([]float64, len(dw))
	for i := range dw {
		w[i] -= dw[i] * lr
	}

	return w, b
}

func (a *App) logisticRegression(inputs [][]float64, y []float64, px, py [][]float64) (w []float64, b float64) {

	for epoch := 1; epoch <= epochs; epoch++ {
		dw, db := dCost(inputs, y, inference(inputs, w, b))
		w, b = gradientDescent(dw, db)
		a.updatePlot(w, b, px, py)
	}

	return w, b
}
