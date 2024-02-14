package main

import (
	"encoding/csv"
	"io"
	"os"
)

// Writes data from .csv file row by row to the sink
func readCSV(path string, sink func(row []string) error) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = ','

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			return err
		}
		if err = sink(record); err != nil {
			return err
		}
	}
	return nil
}
