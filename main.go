package main

import (
	"log"
	"math"
)

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Pow(math.E, -z))
}

func dot(a []float64, b []float64) (dp float64) {
	if len(a) != len(b) {
		log.Fatal("len(a) != len(b)")
	}
	for i := range a {
		dp += a[i] * b[i]
	}
	return
}

func inference(inputs [][]float64, w []float64, b float64) (probs []float64) {
	for _, x := range inputs {
		probs = append(probs, dot(w, x)+b)
	}
	return
}
