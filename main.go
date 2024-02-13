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

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}
func dot(a []float64, b []float64) (dot float64) {
	if len(a) != len(b) {
		fmt.Println(a, b)
		panic("len(a) != len(b)")
	}
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
	}
	return dot
}

func inference(inputs [][]float64, w []float64, b float64) (res []float64) {
	res = make([]float64, len(inputs))
	for i := 0; i < len(inputs); i++ {
		res[i] = sigmoid(dot(inputs[i], w) + b)
	}
	return res
}
func dCost(inputs [][]float64, y, p []float64) (dw []float64, db float64) {
	dw = make([]float64, len(inputs[0]))
	m := len(inputs)
	for i := 0; i < m; i++ {
		for j := 0; j < len(inputs[0]); j++ {
			dw[j] += (inputs[i][j] * (p[i] - y[i])) / float64(m)
		}
		db += (p[i] - y[i]) / float64(m)
	}
	return dw, db
}

func gradientDescent(inputs [][]float64, y, w []float64, alpha, b float64) {
	for i := 0; i < 100; i++ {
		p := inference(inputs, w, b)
		dw, db := dCost(inputs, y, p)
		for j := 0; j < len(w); j++ {
			w[j] -= alpha * dw[j]
		}
		b -= alpha * db
		//fmt.Println(dw, db)
	}
	//fmt.Println(w,b)
}

func accuracy(inputs [][]float64, y []float64, w []float64, b float64) float64 {
	prediction := inference(inputs, w, b)
	var truePos, trueNeg, falsePos, falseNeg float64
	for i := 0; i < len(y); i++ {
		if prediction[i] >= 0.5 {
			if y[i] == 1 {
				truePos++
			} else {
				falsePos++
			}
		} else {
			if y[i] == 0 {
				trueNeg++
			} else {
				falseNeg++
			}
		}
	}
	return (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
}

func split(data [][]string) (xTrain, xTest [][]float64, yTrain, yTest []float64) {
	half := len(data) / 2
	segment := len(data[0])
	xTrain = make([][]float64, half)
	for i := range xTrain {
		xTrain[i] = make([]float64, segment)
	}
	yTrain = make([]float64, half)

	xTest = make([][]float64, half)
	for i := range xTest {
		xTest[i] = make([]float64, segment)
	}
	yTest = make([]float64, half)

	for i, row := range data[:half] {
		for j := 0; j < 2; j++ {
			xTrain[i][j], _ = strconv.ParseFloat(row[j], 64)
		}
		yTrain[i], _ = strconv.ParseFloat(row[2], 64)
	}

	for i, row := range data[half:] {
		for j := 0; j < 2; j++ {
			xTest[i][j], _ = strconv.ParseFloat(row[j], 64)
		}
		yTest[i], _ = strconv.ParseFloat(row[2], 64)
	}
	return xTrain, xTest, yTrain, yTest
}

func main() {
	//reading
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
	xTrain, xTest, yTrain, yTest := split(data)
	w := make([]float64, len(xTrain[0]))
	for i := range w {
		w[i] = rand.Float64()
	}
	b := rand.Float64()
	alpha := 1e-3
	gradientDescent(xTrain, yTrain, w, alpha, b)
	//accuracy(xTrain, yTrain, w, b)
	score := accuracy(xTest, yTest, w, b)
	fmt.Println(score)
}
