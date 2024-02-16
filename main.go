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
	return 1 / (1 + math.Exp(-z))
}

func dot(a []float64, b []float64) float64 {
	var res float64
	if len(a) != len(b) {
		return 0
	}
	for i := 0; i < len(a); i++ {
		res += a[i] * b[i]
	}
	return res
}

func inference(inputs [][]float64, w []float64, b float64) []float64 {
	res := make([]float64, len(inputs))
	for j, x := range inputs {
		f := dot(w, x)
		res[j] = sigmoid(f + b)
	}
	return res
}

func dCost(inputs [][]float64, y, p []float64) (dw []float64, db float64) {
	m := float64(len(inputs))
	dw = make([]float64, len(inputs[0]))
	for i := range inputs {
		for j := range inputs[i] {
			dw[j] += (p[i] - y[i]) * inputs[i][j] / m
		}
		db += (p[i] - y[i]) / m
	}
	return dw, db
}

func gradientDescent(inputs [][]float64, y, w []float64, lr, b float64) ([]float64, float64) {
	for i := 0; i < 5000; i++ {
		p := inference(inputs, w, b)
		dw, db := dCost(inputs, y, p)
		for j := range w {
			w[j] -= lr * dw[j]
		}
		b -= lr * db
	}
	return w, b
}

func accuracy(inputs [][]float64, y []float64, w []float64, b float64) (acc float64) {
	r := inference(inputs, w, b)
	for i := range inputs {
		if y[i] == math.Round(r[i]) {
			acc++
		}
	}
	return acc / float64(len(y))
}

func split(inputs [][]float64, y []float64) (xTrain, xTest [][]float64, yTrain, yTest []float64) {
	size := len(inputs) / 5
	return inputs[size:], inputs[:size], y[size:], y[:size]
}

func shuffle(inputs [][]float64, y []float64) (sInputs [][]float64, sY []float64) {
	ind := rand.Perm(len(inputs))
	sInputs, sY = make([][]float64, len(inputs)), make([]float64, len(y))
	for i, j := range ind {
		sInputs[i], sY[i] = inputs[j], y[j]
	}
	return sInputs, sY
}

func main() {
	data, err := readDataFromCSV("data/exams1.csv")
	if err != nil {
		log.Fatalf("%v", err)
	}
	inputs := make([][]float64, len(data))
	y := make([]float64, len(data))
	for i, student := range data {
		inputs[i] = []float64{student.Exam1, student.Exam2}
		y[i] = float64(student.Accepted)
	}
	w := make([]float64, len(inputs[0]))
	for i := range w {
		w[i] = 1
	}
	lr := 0.01
	b := 0.
	fmt.Println(gradientDescent(inputs, y, w, lr, b))
	sInputs, sY := shuffle(inputs, y)
	fmt.Println(split(sInputs, sY))
}

type Student struct {
	Exam1    float64
	Exam2    float64
	Accepted int
}

func readDataFromCSV(exams1 string) ([]Student, error) {
	file, err := os.Open(exams1)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var students []Student
	for _, record := range records {
		exam1, _ := strconv.ParseFloat(record[0], 64)
		exam2, _ := strconv.ParseFloat(record[1], 64)
		accepted, _ := strconv.Atoi(record[2])

		student := Student{
			Exam1:    exam1,
			Exam2:    exam2,
			Accepted: accepted,
		}
		students = append(students, student)
	}
	return students, nil
}
