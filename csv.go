package main

import (
	"encoding/csv"
	"io"
	"log"
	"os"
	"strconv"
)

const (
	numberOfPoints         = 100
	numberOfTrainingPoints = 80
)

func getPoints() (inputs [][]float64, y []float64) {

	// ############## Opening CSV file ##############

	file, err := os.Open("data/exams1.csv")
	if err != nil {
		log.Fatal(err)
		return nil, nil
	}

	defer file.Close() //Declaring file closure at the end

	return readPoints(file)
}

// Reads CSV file and saves data in [][] of inputs and [] of whether point is true or not
func readPoints(file io.Reader) ([][]float64, []float64) {

	// ############## Declaration ##############

	inputs := make([][]float64, numberOfPoints)
	y := make([]float64, numberOfPoints)

	// ############## Making new reader ##############

	reader := csv.NewReader(file) // new reader
	reader.Comma = ','

	// ############## Reading data ##############

	for row := 0; ; row++ { //until err == io.EOF

		// ##### Reading one record at a time #####

		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		// ##### Distributing record data along variables #####

		inputs[row] = make([]float64, 2) //making slice instance for every record

		for i := 0; i < len(record)-1; i++ { //for every value besides the last one in the row
			inputs[row][i], err = strconv.ParseFloat(record[i], 64) //saving value in input[][]
			if err != nil {
				log.Fatal(err)
			}

		}

		// saving value in y
		y[row], err = strconv.ParseFloat(record[len(record)-1], 64)
		if err != nil {
			log.Fatal(err)
		}

	}

	return inputs, y
}

// Splits the input data set into the Training and Test data sets
func split(inputs [][]float64, y []float64) (xTrain, xTest [][]float64, yTrain, yTest []float64) {

	//Saving training points
	for i := 0; i < numberOfTrainingPoints; i++ { // 80 points
		xTrain = append(xTrain, inputs[i]) //saving X slice with features
		yTrain = append(yTrain, y[i])      //saving Y
	}

	//Saiving test points
	for i := 0; i < numberOfPoints-numberOfTrainingPoints; i++ { // 100-80 = 20 points
		xTest = append(xTest, inputs[i]) //saving X slice with features
		yTest = append(yTest, y[i])      //saving Y
	}

	return xTrain, xTest, yTrain, yTest
}
