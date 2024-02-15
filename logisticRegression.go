package main

import (
	"fmt"
	"math"
)

const (
	epochs = 10000
	lrw    = 0.000001
	lrb    = 5
)

// Takes all point data, calculates logistic regression, calculates accuracy and sends everything for drawing
func (a *App) logisticRegression(xTrain, xTest [][]float64, yTrain, yTest []float64) {

	w := make([]float64, len(xTrain[0])) //declaring w coefficients
	var b float64                        //declaring b coefficient

	for epoch := 1; epoch <= epochs; epoch++ { // for every epoch
		w, b = gradientDescent(w, b, xTrain, yTrain) // adjusting all coefficients

		if epoch%100 == 0 { //every 100th epoch{
			fmt.Println("Epoch", epoch, "w1:", w[0], "| w2:", w[1], "| b:", b) // coefficient debug
			fmt.Println("Accuracy:", accuracy(xTest, yTest, w, b), "\n")       //Getting accuracy of the trained logistic regression
		}

		a.updatePlot(w, b, xTrain, yTrain) //recreating plot with new values
		//time.Sleep(time.Millisecond)       //delay to monitor the updates
	}

}

// adjusting coefficients for w1*x1 + w2*x2 + ... + wn*xn + b
func gradientDescent(w []float64, b float64, input [][]float64, y []float64) ([]float64, float64) {

	predictions := inference(input, w, b)  // getting current predictions
	dw, db := dCost(input, y, predictions) // getting current gradients

	for feature := range w { // for every feature
		w[feature] -= dw[feature] * lrw // adjusting w coefficient
	}
	b -= db * lrb // adjusting b coefficient

	return w, b // returning adjusted coefficients
}

// dCost returns gradients for all features and bias
// input [][]float64 - are points with all the features
// y []float64 is whether some point is true or not
// dw - gradients for all the features
// db - gradient for ... eh..
func dCost(inputs [][]float64, y []float64, predictions []float64) (dw []float64, db float64) {
	/*
		Finds dw, db via mse
		dw[i] = ((1/m)*(p(x[i])-y[i])*x[i])
		db = ((1/m)*(p(x[i])-y[i])*x[i])
	*/

	m := float64(len(inputs))
	dw = make([]float64, len(inputs[0])) //initializing lenght of the dw slice

	for i, point := range inputs { //for each point
		for feature := range point { //for each feature
			dw[feature] += (1 / m) * (predictions[i] - y[i]) * point[feature] //calculate feature gradient by formula of mse
		}
		db += (1 / m) * (predictions[i] - y[i]) // calculating bias gradient by mse formula
	}
	return dw, db
}

// a.k.a. prediction (for all points)
func inference(inputs [][]float64, w []float64, b float64) []float64 {
	p := make([]float64, len(inputs)) //declaring prediction slice
	for i := range inputs {           //for every point
		p[i] = g(dot(w, inputs[i]) + b) //w - weights for each feature | input[i] - concrete point
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
