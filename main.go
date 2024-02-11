package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
)

const (
	epochs       = 100
	learningRate = 0.03
)

func main() {
	file, err := os.Open("data/exams1.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	data, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	// variables
	inputs := make([][]float64, 100)
	for i := range inputs {
		inputs[i] = make([]float64, 2)
	}
	y := make([]float64, 100)
	for i, row := range data {
		for j := 0; j < 2; j++ {
			inputs[i][j], _ = strconv.ParseFloat(row[j], 64)
		}
		y[i], _ = strconv.ParseFloat(row[2], 64)
	}
	w := make([]float64, len(inputs[0]))
	for i := range w {
		w[i] = rand.Float64()
	}

	b := 0.
	gradient(inputs, y, w, b)
}

func gradient(inputs [][]float64, y, w []float64, b float64) {
	for i := 0; i <= epochs; i++ {
		fxi := inference(inputs, w, b)
		dw, db := deratives(inputs, y, fxi)
		for j := 0; j < len(w); j++ {
			w[j] = w[j] - learningRate*dw[j]
		}
		b = b - learningRate*db

		if i%10 == 0 {
			fmt.Printf("Epoch nuber: %d\nw: %f\nb: %f\n", i, w, b)
		}
	}
}

func deratives(inputs [][]float64, y, fxi []float64) (dw []float64, db float64) {
	dw = make([]float64, len(inputs[0]))
	n := len(inputs)
	for i := 0; i < n; i++ {
		for j := 0; j < len(inputs[0]); j++ {
			dw[j] += ((fxi[i] - y[i]) * inputs[i][j]) / float64(n)
		}
		db += (fxi[i] - y[i]) / float64(n)
	}
	return dw, db
}

func inference(inputs [][]float64, w []float64, b float64) []float64 {
	var predictions []float64
	for _, x := range inputs {
		predictions = append(predictions, linearModel(x, w, b))
	}
	return predictions
}

func sigmoid(z float64) float64 { return 1 / (1 + math.Pow(math.E, -z)) }

func linearModel(x, w []float64, b float64) (res float64) { return sigmoid(dot(x, w) + b) }

func dot(a []float64, b []float64) (res float64) {
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}

// func loss(p, y float64) float64 { return -y*math.Log(p) - (1-y)*math.Log(1-p) }

// func cost(n int, y, fxi []float64) (res float64) {
// 	for i := 0; i < n; i++ {
// 		res += loss(fxi[i], y[i])
// 	}
// 	return res / float64(n)
// }
