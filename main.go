package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
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

func shuffle(inputs [][]float64, y []float64) {
	for i := range inputs {
		if len(inputs)-i == 0 {
			return
		}
		t := rand.Intn(len(inputs) - i)
		inputs[i], inputs[i+t] = inputs[i+t], inputs[i]
		y[i], y[i+t] = y[i+t], y[i]
	}
}

func split(inputs [][]float64, y []float64) (xTrain, xTest [][]float64, yTrain, yTest []float64) {
	for i := range inputs {
		if i%4 == 0 {
			xTest = append(xTest, inputs[i])
			yTest = append(yTest, y[i])
		} else {
			xTrain = append(xTrain, inputs[i])
			yTrain = append(yTrain, y[i])

		}
	}
	return
}

func Round(a float64) float64 {
	if a < 0.5 {
		return 0
	}
	if a >= 0.5 {
		return 1
	}
	panic("Your computer is broken")
}

func accuracy(inputs [][]float64, y []float64, w []float64, b float64) float64 {
	success := 0
	res := inference(inputs, w, b)
	for i := range res {
		if Round(res[i]) == y[i] {
			success++
		}
	}
	return float64(success) / float64(len(res))
}

func main() {
	dataset := make([][]float64, 0)
	res := make([]float64, 0)
	file, err := os.Open("data/exams1.csv")

	if err != nil {
		fmt.Println(err)
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Println(err)
			continue
		}
		a, _ := strconv.ParseFloat(record[0], 64)
		b, _ := strconv.ParseFloat(record[1], 64)
		c, _ := strconv.ParseFloat(record[2], 64)
		dataset = append(dataset, []float64{a, b})
		res = append(res, c)
	}
	shuffle(dataset, res)

	xTrain, xTest, yTrain, yTest := split(dataset, res)
	m := model{[]float64{0, 0}, 0}
	m.Train(xTrain, yTrain, 1e-3, 1e-2, 10000000, xTest, yTest)
}
