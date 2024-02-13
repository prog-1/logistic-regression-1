package main

import (
	"math"
	"testing"
)

func TestSigmoid(t *testing.T) {
	for _, tc := range []struct {
		z    float64
		want float64
	}{
		{-100, 0.000001},
		{-1, 0.2689414},
		{0, 0.5},
		{1, 0.7310585},
		{2, 0.8807970},
		{100, 0.9999999},
	} {
		if got := sigmoid(tc.z); math.Abs(tc.want-got) > 1e-6 {
			t.Errorf("got = %v, want = %v", got, tc.want)
		}
	}
}

func TestDot(t *testing.T) {
	for _, tc := range []struct {
		a    []float64
		b    []float64
		want float64
	}{
		{[]float64{0, 0}, []float64{0, 0}, 0},
		{[]float64{1, 1}, []float64{0, 0}, 0},
		{[]float64{1, 2}, []float64{3, 0}, 3},
		{[]float64{1, -1}, []float64{1, 1}, 0},
		{[]float64{-1, -1}, []float64{-1, -1}, 2},
		{[]float64{-10, 100}, []float64{100, 100}, 9000},
	} {
		if got := dot(tc.a, tc.b); math.Abs(tc.want-got) > 1e-6 {
			t.Errorf("dot(%v, %v) = %v, want = %v", tc.a, tc.b, got, tc.want)
		}
	}
}
