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
	epochs       = 1000
	learningRate = 0.0001
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

	xTrain, xTest, yTrain, yTest := split(inputs, y)
	wTrained, bTrained := gradient(xTrain, yTrain, w, 0.)
	fmt.Println("Final accuracy:", accuracy(xTest, yTest, wTrained, bTrained))
}

func gradient(inputs [][]float64, y, w []float64, b float64) ([]float64, float64) {
	for i := 0; i <= epochs; i++ {
		fxi := inference(inputs, w, b)
		dw, db := deratives(inputs, y, fxi)
		cost := cost(len(inputs), fxi, y)
		for j := 0; j < len(w); j++ {
			w[j] = w[j] - learningRate*dw[j]
		}
		b = b - learningRate*db
		if i%100 == 0 {
			fmt.Printf("Epoch nuber: %d\ndw: %f\ndb: %f\ncost: %f\n", i, dw, db, cost)
		}
	}
	return w, b
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

func loss(p, y float64) float64 {
	if y == 1 {
		return -math.Log(p)
	} else {
		return -math.Log(1 - p)
	}
}

func cost(n int, fxi, y []float64) (res float64) {
	for i := 0; i < n; i++ {
		l := loss(fxi[i], y[i])
		// if math.IsInf(l,1){
		// 	fmt.Println(fxi[i], y[i])
		// }
		res += l
	}
	return res / float64(n)
}

func split(inputs [][]float64, y []float64) (xTrain, xTest [][]float64, yTrain, yTest []float64) {
	xTrain, xTest = inputs[:len(inputs)*9/10], inputs[len(inputs)*9/10:]
	yTrain, yTest = y[:len(y)*9/10], y[len(y)*9/10:]
	return xTrain, xTest, yTrain, yTest
}

func accuracy(inputs [][]float64, y []float64, w []float64, b float64) float64 {
	var res float64
	for i, x := range inputs {
		if y[i] == math.Round(sigmoid(dot(x, w)+b)) {
			res++
		}
	}
	return res / float64(len(y)) * 100
}
