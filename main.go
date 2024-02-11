package main

import (
	"log"

	"github.com/hajimehoshi/ebiten"
)

func main() {
	//####################### Points #########################

	inputs, y := getPoints()
	px, py := groupPoints(inputs, y)

	//####################### Ebiten ####################################

	//Window
	ebiten.SetWindowSize(sW, sH)
	ebiten.SetWindowTitle("Logistic Regression")

	//App instance
	a := NewApp(sW, sH)

	//Starting linear regression in another thread
	go func() {
		a.logisticRegression(inputs, y, px, py)
	}()

	//Running game
	if err := ebiten.RunGame(a); err != nil {
		log.Fatal(err)
	}
}
