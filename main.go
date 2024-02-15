package main

import (
	"fmt"
	"log"
)

const (
	epochCount = 1e6
	lrw, lrb   = 1e-3, 0.5
)

func main() {
	xs, ys := ReadExams1()
	// trainingInputScatters, err := trainingInputScatters(PosXS, NegXS)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// var learningRates WeightRelated = WeightRelated{
	// 	ws: [argumentCount]float64{1e-2, 1e-2},
	// 	b:  5e-3,
	// }

	// img := make(chan *image.RGBA, 1)
	// go func() {
	sink := func(epoch int, w, dw []float64, b, db float64) {
		if epoch%1e4 != 0 {
			return
		}
		// select {
		// case img <- Plot(trainingInputScatters[0], trainingInputScatters[1],
		// 	/*TODO: Pass decision boundary function*/):
		// default:
		// }
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

	// }()

	// if err := ebiten.RunGame(&App{Img: img}); err != nil {
	// 	log.Fatal(err)
	// }
}
