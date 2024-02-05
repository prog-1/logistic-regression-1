package main

import (
	"testing"
)

func TestSigmoid(t *testing.T) {
	for _, tc := range []struct {
		z    float64
		want float64
	}{
		{-1, 0.2689414213699951},
		{0, 0.5},
		{1, 0.7310585786300049},
		{2, 0.8807970779778823},
		{3, 0.9525741268224334},
	} {
		if got := sigmoid(tc.z); got != tc.want {
			t.Errorf("got = %v, want = %v", got, tc.want)
		}
	}
}

func TestDot(t *testing.T) {
	for _, tc := range []struct {
		a, b []float64
		want float64
	}{
		{[]float64{0, 0, 0}, []float64{0, 0, 0}, 0},
		{[]float64{1, 0, 0}, []float64{0, 0, 0}, 0},
		{[]float64{1, 0, 0}, []float64{1, 0, 0}, 1},
		{[]float64{-1, 0, 0}, []float64{0, 0, 0}, 0},
		{[]float64{-1, 0, 0}, []float64{1, 0, 0}, -1},
		{[]float64{-1, 0, 0}, []float64{-1, 0, 0}, 1},
		{[]float64{1, 0, 0}, []float64{0, 1, 0}, 0},
		{[]float64{1, 2, 3}, []float64{4, 5, 6}, 32},
	} {
		if got := dot(tc.a, tc.b); got != tc.want {
			t.Errorf("Dot(%v)(%v) got = %v, want = %v", tc.a, tc.b, got, tc.want)
		}
	}
}
