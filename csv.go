package main

import (
	"encoding/csv"
	"io"
	"log"
	"os"
	"strconv"
)

const (
	numberOfPoints = 100
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

// // Splits [][] slice into 2x [] slices
// func split2DSlice(s [][]float64) (s1, s2 []float64) {
// 	for row := range s { // for every slice (row)
// 		s1 = append(s1, s[row][0])
// 		s2 = append(s2, s[row][1])
// 		//slight hardcode over here
// 	}
// 	return s1, s2
// }
