package main

import (
	"math"
	"math/rand"
)

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func dot(a, b []float64) (res float64) {
	if len(a) != len(b) {
		panic("len(a) != len(b)")
	}
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}

func prediction(x, w []float64, b float64) float64 {
	return sigmoid(dot(x, w) + b)
}

func inference(inputs [][]float64, w []float64, b float64) (probabilities []float64) {
	if len(inputs[0]) != len(w) {
		panic("len(inputs[0]) != len(w)")
	}

	for _, x := range inputs {
		probabilities = append(probabilities, prediction(x, w, b))
	}
	return probabilities
}

func dCost(inputs [][]float64, labels, y []float64) (dw []float64, db float64) {
	dw = make([]float64, len(inputs[0]))
	var diff float64
	m := float64(len(inputs))
	for i := range inputs {
		diff = y[i] - labels[i]
		for j := range dw {
			dw[j] += 1 / m * diff * inputs[i][j]
		}
		db += 1 / m * diff
		// Usually do: diff = label - prediction; derivative -= ...
	}
	return dw, db
}

func train(epochCount int, inputs [][]float64, ys []float64, lrw, lrb float64, sink func(epoch int, w, dw []float64, b, db float64)) (w []float64, b float64, err error) {
	if len(inputs) < 1 {
		// return weights, errors.New("no training examples provided")
		panic("no training examples provided")
	}

	w = make([]float64, len(inputs[0]))
	for epoch := 0; epoch < epochCount; epoch++ {
		dw, db := dCost(inputs, ys, inference(inputs, w, b))

		// Adjusting weights
		b -= db * lrb
		for i := range w {
			w[i] -= dw[i] * lrw
		}

		sink(epoch, w, dw, b, db)
	}

	return w, b, nil
}

func decisionBoundaryLinearFunc(w []float64, b float64) func(float64) float64 {
	if len(w) != 2 {
		panic("more than 2 parameters in model")
	}
	return func(x float64) float64 {
		return -(w[0]*x + b) / w[1]
	}
}

// Takes the first ratio(%) as the testing set
func split(inputs [][]float64, y []float64, ratio int) (xTrain, xTest [][]float64, yTrain, yTest []float64) {
	testElCount := len(inputs) * ratio / 100
	return inputs[testElCount:], inputs[:testElCount], y[testElCount:], y[:testElCount]
}

func accuracy(inputs [][]float64, y []float64, w []float64, b float64) float64 {
	var correctCount float64
	for i, x := range inputs {
		if y[i] == math.Round(prediction(x, w, b)) {
			correctCount++
		}
	}
	return correctCount / float64(len(inputs))
}

// Shuffles input in place
func shuffleInput(inputs [][]float64, y []float64) {
	if len(inputs) != len(y) {
		panic("len(inputs) != len(y)")
	}

	rand.Shuffle(len(inputs), func(i, j int) {
		inputs[i], y[i] = inputs[j], y[j]
	})
}
