package main

import (
	"log"

	"github.com/hajimehoshi/ebiten/v2"
)

func main() {

	//####################### Points #########################

	inputs, y := getPoints() //gettings data sets

	xTrain, xTest, yTrain, yTest := split(inputs, y) //splitting data sets

	//####################### Ebiten #########################

	//Window
	ebiten.SetWindowSize(sW, sH)
	ebiten.SetWindowTitle("Linear Regression")

	//App instance
	a := NewApp(sW, sH)

	//####################### Logistic Regression #########################

	go func() { //Starting logistic regression in another thread
		a.logisticRegression(xTrain, xTest, yTrain, yTest)
	}()

	//####################### Ebiten #########################

	//Running game
	if err := ebiten.RunGame(a); err != nil {
		log.Fatal(err)
	}

}
