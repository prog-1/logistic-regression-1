package main

import "fmt"

func main() {

	/*
		1. Get input & y from CSV and also x1,x2 of the same inputs for drawing
		2. Give them to logistic regression
		3. Group as columns

	*/

	//####################### Points #########################

	inputs, y := getPoints() //gettings point data

	//####################### Logistic Regression #########################

	w, b := logisticRegression(inputs, y) // making logistic regression

	//####################### Debug #########################

	for i := range w {
		fmt.Print("w", i, ": ", w[i], "|")
	}
	fmt.Print(" B: ", b, "\n")

}
