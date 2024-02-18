package main

import (
	"fmt"
	"image"
	"image/color"
	"log"

	"github.com/hajimehoshi/ebiten/v2"
	"gonum.org/v1/plot/plotter"
)

const (
	epochCount                = 1e6
	lrw, lrb                  = 1e-3, 0.5
	x1min, x1max              = 0, 100
	screenHeight, screenWidth = 720, 480
)

func main() {
	inputs, ys := ReadExams1()
	posScatter, negScatter, err := trainingInputScatters(inputs, ys, color.RGBA{0, 255, 0, 255}, color.RGBA{255, 0, 0, 255})
	if err != nil {
		log.Fatal(err)
	}

	img := make(chan *image.RGBA, 1)
	sink := func(epoch int, w, dw []float64, b, db float64) {
		if epoch%1e4 != 0 {
			return
		}
		dbf, err := plotter.NewLine(plotter.XYs{{X: x1min, Y: decisionBoundaryFunc(w, b)(x1min)}, {X: x1max, Y: decisionBoundaryFunc(w, b)(x1max)}})
		if err != nil {
			log.Fatal(err)
		}
		select {
		case img <- Plot(posScatter, negScatter, dbf):
		default:
		}
		fmt.Printf("Epoch: %v\n\n", epoch)
		fmt.Printf("Derivatives:\nws = %v\nb = %v\n\n", dw, db)
		fmt.Println()
	}
	go func() {
		if w, b, err := train(epochCount, inputs, ys, lrw, lrb, sink); err != nil {
			log.Fatal(err)
		} else {
			fmt.Printf("\n\nWeights:\nws = %v\nb = %v\n\n", w, b)
			fmt.Printf("\n\nAccuracy: %v\n\n", accuracy(inputs, ys, w, b))
		}
	}()

	if err := ebiten.RunGame(&App{Img: img}); err != nil {
		log.Fatal(err)
	}
}
