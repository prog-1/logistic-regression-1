package main

import (
	"math"
	"time"
)

const (
	epochs = 10000
	lr     = 0.001
)

// takes all point data, calculates logistic regression and sends them for drawing
func logisticRegression(input [][]float64, y []float64) {

	w := make([]float64, len(input[0])) //declaring w coefficients
	var b float64                       //declaring b coefficient

	for epoch := 1; epoch <= epochs; epoch++ { // for every epoch
		w, b := gradientDescent(w, b, input, y) // adjusting all coeficients
		time.Sleep(time.Millisecond)            //delay to monitor the updates
	}
	//draw somehow
}

// adjusting coefficients for w1*x1 + w2*x2 + ... + wn*xn + b
func gradientDescent(w []float64, b float64, input [][]float64, y []float64) ([]float64, float64) {

	predictions := inference(input, w, b)  // get current predictions
	dw, db := dCost(input, y, predictions) // get current gradients

	for feature := range w { // for every feature
		w[feature] -= dw[feature] * lr // adjust w coefficient
	}
	b -= db * lr // adjust b coefficient

	return w, b // returning adjusted coefficients
}

// dCost returns gradients for all features and bias
// input [][]float64 - are points with all the features
// y []float64 is whether some point is true or not
// dw - gradients for all the features
// db - gradient for ... eh..
func dCost(input [][]float64, y []float64, predictions []float64) (dw []float64, db float64) {
	/*
		Finds dw, db via mse
		dw[i] = ((1/m)*(p(x[i])-y[i])*x[i])
		db = ((1/m)*(p(x[i])-y[i])*x[i])
	*/
	m := float64(len(input)) //number of points

	for i, point := range input { //for each point
		for feature := range point { //for each feature
			dw[feature] += (1 / m) * (predictions[i] - y[i]) * point[feature] //calculate feature gradient by formula of mse
		}
		db += (1 / m) * (predictions[i] - y[i]) // calculating bias gradient by mse formula
	}
	return dw, db
}

// a.k.a. prediction (for all points)
func inference(input [][]float64, w []float64, b float64) (p []float64) {
	for i := range input { //for every point
		p[i] = g(dot(w, input[i]) + b) //w - weights for each feature | input[i] - concrete point
	}
	return p
}

// Dot product
func dot(a, b []float64) (res float64) {
	for i := range a { // for each dimention
		res += a[i] * b[i] // multiply each dimention and sum up
	}
	return res
}

// Sigmoid function
func g(z float64) float64 {
	return 1 / (1 + math.Pow(math.E, -1*z))
}
