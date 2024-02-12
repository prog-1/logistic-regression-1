package main

import "log"

type model struct {
	w []float64
	b float64
}

func (m *model) Train(x [][]float64, y []float64, lrk, lrb float64, epochs int) {
	for i := 0; i < epochs; i++ {
		t := inference(x, m.w, m.b)
		dw, db := dCost(x, y, t)
		for j := range dw {
			m.w[j] -= lrk * dw[j]
		}
		m.b -= lrb * db
		log.Printf(`Epoch: %d/%d, 
			dk: %v, db: %v
			`, i, epochs, dw, db)
	}
}
