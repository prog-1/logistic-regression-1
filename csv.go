package main

import (
	"encoding/csv"
	"io"
	"log"
	"os"
	"strconv"
)

func getPoints() (inputs [][]float64, y []float64) {

	//Opening CSV file
	file, err := os.Open("exams1.csv")
	if err != nil {
		log.Fatal(err)
		return nil, nil
	}

	//Declaring file closure at the end
	defer file.Close()

	return readPoints(file)
}

func readPoints(file io.Reader) (inputs [][]float64, y []float64) {

	reader := csv.NewReader(file) // new reader
	reader.Comma = ','

	for r := 0; ; r++ { //until err != io.EOF

		//Reading one record at a time
		record, err := reader.Read()
		if err != io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		//x1 & x2
		for i := 0; i < len(record)-1; i++ { //every column besides the last one
			inputs[r][i], err = strconv.ParseFloat(record[i], 64) //save in variable
			if err != nil {
				log.Fatal(err)
			}
		}

		//y
		y[r], err = strconv.ParseFloat(record[len(record)-1], 64)
		if err != nil {
			log.Fatal(err)
		}

	}
	return inputs, y
}

func groupPoints(inputs [][]float64, y []float64) ([][]float64, [][]float64) {
	px := make([][]float64, 2)
	py := make([][]float64, 2)

	for i := 0; i < len(px); i++ { //for every point
		if y[i] == 0 { // if it is false
			px[0] = append(px[0], inputs[i][0])
			py[0] = append(py[0], inputs[i][1])
		} else { // if it is true
			px[1] = append(px[1], inputs[i][0])
			py[1] = append(py[1], inputs[i][1])
		}
	}
	return px, py

}
