package main

import (
	"log"
	"strconv"
)

type Student struct {
	examScores [2]float64
	accepted   bool
}

type Error struct {
	message string
}

// Reads data from data/exams1.csv
// Handles errors on its own with log.Fatal
func readStudentsFromCSV() (students []Student) {
	appendStudent := func(row []string) {
		if len(row) != 3 {
			log.Fatal("readStudentsFromCSV: Row length != 3")
		}

		var student Student

		stringToFloat := func(str string) float64 {
			output, err := strconv.ParseFloat(str, 64)
			if err != nil {
				log.Fatal(err)
			}
			return output
		}
		stringToBool := func(str string) bool {
			output, err := strconv.ParseBool(str)
			if err != nil {
				log.Fatal(err)
			}
			return output
		}

		student.examScores[0] = stringToFloat(row[0])
		student.examScores[1] = stringToFloat(row[1])
		student.accepted = stringToBool(row[2])
		students = append(students, student)
	}
	if err := readCSV("data/exams1.csv", appendStudent); err != nil {
		log.Fatal(err)
	}
	return students
}
