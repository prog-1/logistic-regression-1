package main

import (
	"image"
	"log"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
)

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-1*z))
}

func main() {
	ebiten.SetWindowSize(640, 480)
	ebiten.SetWindowTitle("Logistic regression")

	img := make(chan *image.RGBA, 1)
	go func() {
		p := Plot(-10, 10, 0.1, sigmoid)
		x := 0.0
		img <- p(x)
		// for i := 0; i < 50; i++ {
		// 	time.Sleep(30 * time.Millisecond)
		// 	x -= dsigmoid(x) * 0.1
		// 	img <- p(x)
		// }
	}()

	if err := ebiten.RunGame(&App{Img: img}); err != nil {
		log.Fatal(err)
	}
}
