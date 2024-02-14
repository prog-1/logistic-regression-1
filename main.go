package main

import (
	"errors"
	"fmt"
	"image"
	"log"
)

// sinkingCondition() defines in what case sink() is executed
func train(xs Inputs, ys []float64, sinkingCondition func(epoch int) bool, sink func(epoch int, weights, derivatives WeightRelated)) (weights WeightRelated, err error) {
	if len(xs) < 1 {
		return weights, errors.New("No training examples provided")
	}
	// m := len(xs)    // number of training examples
	// n := len(xs[0]) // number of features

	return weights, nil
}

func main() {
	xs, ys := readExams1()
	img := make(chan *image.RGBA, 1)
	PosXS, NegXS, err := splitTrainingSet(xs, ys)
	if err != nil {
		log.Fatal(err)
	}
	trainingInputScatters, err := trainingInputScatters(PosXS, NegXS)
	if err != nil {
		log.Fatal(err)
	}

	if weights, err := train(xs, ys, func(epoch int) bool { return epoch%1000 == 0 }, func(epoch int, weights, derivatives WeightRelated) {
		select {
		case img <- Plot(trainingInputScatters[0], trainingInputScatters[1],
			/*TODO: Pass categorisation division function*/):
		default:
		}
		fmt.Printf("Epoch: %v\n\n", epoch)
		fmt.Printf("Weights:\nw0 = %v, ws = %v\nb = %v, bws = %v\n\n", weights.ws, weights.ws, weights.b, weights.bws)
		fmt.Printf("Derivatives:\nw0 = %v, ws = %v\nb = %v, bws = %v\n\n", derivatives.w0, derivatives.ws, derivatives.b, derivatives.bws)
		fmt.Println()
	}); err != nil {
		log.Fatal(err)
	} else {
		fmt.Println(weights)
	}
}
