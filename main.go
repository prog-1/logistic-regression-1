package main

import "fmt"

const (
	epochCount                = 1e5
	lrw, lrb                  = 1e-3, 0.5
	x1min, x1max              = 0, 100
	screenWidth, screenHeight = 640, 480
	testingSetRatio           = 20 // Percentage of the dataset to be used for testing
)

func main() {
	inputs, _, err := readCSV("C:/Common/Projects/School/logistic-regression-1/data/exams1.csv", 2)
	if err != nil {
		panic(err)
	}
	for i := 0; i < 100; i++ {
		for j := 0; j < 2; j++ {
			fmt.Printf("%v ", inputs.At(i, j))
		}
		fmt.Println()
	}
}

// func main() {
// 	inputs, ys, err := readCSV("C:/Common/Projects/School/logistic-regression-1/data/exams1.csv", 2)
// 	if err != nil {
// 		panic(err)
// 	}
// 	shuffleInput(inputs, ys)
// 	xTrain, xTest, yTrain, yTest := split(inputs, ys, testingSetRatio)

// 	posScatter, negScatter, err := trainingInputScatters(xTrain, yTrain, color.RGBA{0, 255, 0, 255}, color.RGBA{255, 0, 0, 255})
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	imgChannel := make(chan *image.RGBA, 1)
// 	sink := func(epoch int, w, dw []float64, b, db float64) {
// 		if len(w) != len(dw) {
// 			panic("len(w) != len(dw)")
// 		}
// 		// dbf is the decision boundary function
// 		dbf, err := plotter.NewLine(plotter.XYs{{X: x1min, Y: decisionBoundaryLinearFunc(w, b)(x1min)}, {X: x1max, Y: decisionBoundaryLinearFunc(w, b)(x1max)}})
// 		if err != nil {
// 			log.Fatal(err)
// 		}

// 		legend := fmt.Sprintf("Accuracy: %v", accuracy(inputs, ys, w, b))
// 		select {
// 		case imgChannel <- Plot(legend, posScatter, negScatter, dbf):
// 		default:
// 		}
// 		if epoch%1e4 == 0 {
// 			fmt.Printf("Epoch: %v\n\n", epoch)
// 			fmt.Printf("Derivatives:\nws = %v\nb = %v\n\n", dw, db)
// 			fmt.Println()
// 		}
// 	}
// 	go func() {
// 		if w, b, err := train(epochCount, xTrain, yTrain, lrw, lrb, sink); err != nil {
// 			log.Fatal(err)
// 		} else {
// 			fmt.Printf("\n\nWeights:\nws = %v\nb = %v\n\n", w, b)
// 			fmt.Printf("\n\nAccuracy: %v\n\n", accuracy(xTest, yTest, w, b))
// 		}
// 	}()

// 	if err := ebiten.RunGame(&App{Img: imgChannel}); err != nil {
// 		log.Fatal(err)
// 	}
// }
