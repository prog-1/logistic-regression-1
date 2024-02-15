package main

import (
	"fmt"
	"image"
	"image/color"
	"log"

	"github.com/hajimehoshi/ebiten/v2"
)

const (
	epochCount = 1e6
	lrw, lrb   = 1e-3, 0.5
	screenHeight, screenWidth = 720, 480
)

func main() {
	xs, ys := ReadExams1()
	PosScatter, NegScatter, err := trainingInputScatters(xs, ys, color.RGBA{0, 255, 0, 255}, color.RGBA{255, 0, 0, 255})
	if err != nil {
		log.Fatal(err)
	}

	img := make(chan *image.RGBA, 1)
	go func() {
	sink := func(epoch int, w, dw []float64, b, db float64) {
		if epoch%1e4 != 0 {
			return
		}
		select {
		case img <- Plot(PosScatter, NegScatter,
			/*TODO: Pass decision boundary function*/):
		default:
		}
		fmt.Printf("Epoch: %v\n\n", epoch)
		// fmt.Printf("Weights:\nws = %v\nb = %v\n\n", w, b)
		fmt.Printf("Derivatives:\nws = %v\nb = %v\n\n", dw, db)
		fmt.Println()
	}

	if w, b, err := train(epochCount, xs, ys, lrw, lrb, sink); err != nil {
		log.Fatal(err)
	} else {
		fmt.Printf("Weights:\nws = %v\nb = %v\n\n", w, b)
	}

	}()

	if err := ebiten.RunGame(&App{Img: img}); err != nil {
		log.Fatal(err)
	}
}
