package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// Writes data from .csv file row by row to the sink
func readCSV(path string, paramCount int) (x *mat.Dense, y *mat.Dense, err error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open %q", path)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = paramCount + 1
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read the file %q", path)
	}

	inputs := make([]float64, len(records)*paramCount)
	labels := make([]float64, len(records))
	for i, record := range records {
		for j := 0; j < paramCount; j++ {
			inputs[i*paramCount+j], err = strconv.ParseFloat(record[j], 64)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to parse parameter %q", record[j])
			}
		}
		label, err := strconv.Atoi(record[paramCount])
		if err != nil {
			return nil, nil, fmt.Errorf("failed to parse label %q", record[paramCount])
		}
		labels[i] = float64(label)
	}

	x = mat.NewDense(len(records), paramCount, inputs)
	y = mat.NewDense(len(records), 1, labels)
	return x, y, nil
}
