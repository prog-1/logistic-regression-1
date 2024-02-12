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
	return
}

func inference(inputs [][]float64, w []float64, b float64) (res []float64) {
	res = make([]float64, len(inputs))
	for i := range inputs {
		res[i] = sigmoid(dot(inputs[i], w) + b)
	}
	return
}

func dCost(inputs [][]float64, y, p []float64) (dw []float64, db float64) {
	dw = make([]float64, len(inputs[0]))
	for i := range inputs {
		for j := range inputs[i] {
			dw[j] += ((p[i] - y[i]) * inputs[i][j]) / float64(len(inputs))
		}
		db += (p[i] - y[i]) / float64(len(inputs))
	}
	return
}

func main() {
	m := model{[]float64{0, 0, 0}, 0}
	m.Train([][]float64{
		[]float64{1, 2, 3},
		[]float64{1, -2, 3},
		[]float64{-1, 2, -3},
	}, []float64{0.8021838885585817481543435915519132375077833237084304443062099967, 0.6456563062257954529091106364118829640767984024588060872071562097, 0.3543436937742045470908893635881170359232015975411939127928437902}, 1e-4, 1e-4, 1000000)
}
