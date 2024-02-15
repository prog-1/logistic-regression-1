package main

import (
	"fmt"
	"log"
)

const (
	epochCount   = 1e5
	learningRate = 1e-3
)

// sinkingCondition() defines in what case sink() is executed
func train(epochCount int, xs Inputs, ys []float64, learningRate float64, sink func(epoch int, weights, derivatives WeightRelated)) (weights WeightRelated, err error) {
	if len(xs) < 1 {
		// return weights, errors.New("no training examples provided")
		panic("no training examples provided")
	}

	// weights = RandWeights()
	var derivatives WeightRelated
	for epoch := 0; epoch < epochCount; epoch++ {
		derivatives = dCost(xs, ys, inference(xs, weights))
		weights.adjustWeights(&derivatives, learningRate)
		sink(epoch, weights, derivatives)
	}

	return weights, nil
}

func main() {
	xs, ys := readExams1()
	// PosXS, NegXS, err := splitTrainingSet(xs, ys)
	// if err != nil {
	// 	log.Fatal(err)
	// }
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
	if weights, err := train(epochCount, xs, ys, learningRate, func(epoch int, weights, derivatives WeightRelated) {
		if epoch%1e4 != 0 {
			return
		}
		// select {
		// case img <- Plot(trainingInputScatters[0], trainingInputScatters[1],
		// 	/*TODO: Pass decision boundary function*/):
		// default:
		// }
		fmt.Printf("Epoch: %v\n\n", epoch)
		// fmt.Printf("Weights:\nws = %v\nb = %v\n\n", weights.ws, weights.b)
		fmt.Printf("Derivatives:\nws = %v\nb = %v\n\n", derivatives.ws, derivatives.b)
		fmt.Println()
	}); err != nil {
		log.Fatal(err)
	} else {
		fmt.Printf("Weights:\nws = %v\nb = %v\n\n", weights.ws, weights.b)
	}

	// }()

	// if err := ebiten.RunGame(&App{Img: img}); err != nil {
	// 	log.Fatal(err)
	// }
}
