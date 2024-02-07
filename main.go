package main

import "math"

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

func main() {

}
