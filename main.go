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

func accuracy(Trainputs, Testputs [][]float64, Trainy, Testy []float64, w []float64, b float64) (accuracy float64) {
	newTesty := make([]float64, len(Testputs))
	for i := 0; i < len(Testputs); i++ {
		for j := 0; j < len(Testputs[0]);j++{
			newTesty[i] += w[i] * Testputs[i][j] + b
		}
		fmt.Println(newTesty)
	}
	return 0.0
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
	half := len(data) / 2
	fmt.Println(half)

	Trainputs := make([][]float64, half)
	for i := range Trainputs {
		Trainputs[i] = make([]float64, 2)
	}
	Trainy := make([]float64, 100)

	Testputs := make([][]float64, half)
	for i := range Testputs {
		Testputs[i] = make([]float64, 2)
	}
	Testy := make([]float64, 100)

	for i, row := range data[:half] {
		for j := 0; j < 2; j++ {
			Trainputs[i][j], _ = strconv.ParseFloat(row[j], 64)
		}
		Trainy[i], _ = strconv.ParseFloat(row[2], 64)
	}

	for i, row := range data[half:] {
		for j := 0; j < 2; j++ {
			Testputs[i][j], _ = strconv.ParseFloat(row[j], 64)
		}
		Testy[i], _ = strconv.ParseFloat(row[2], 64)
	}

	w := make([]float64, len(Trainputs[0]))
	for i := range w {
		w[i] = rand.Float64()
	}
	b := rand.Float64()
	alpha :=1e-3
	gradientDescent(Trainputs,Trainy, w, alpha, b)
	accuracy(Trainputs, Testputs, Trainy, Testy, w, b)
}
