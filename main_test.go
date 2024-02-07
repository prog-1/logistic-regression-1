package main

import (
	"math"
	"testing"
)

func TestSigmoid(t *testing.T) {
	for num, tc := range []struct {
		input, want float64
	}{
		{-1, 0.2689414213699951},
		{0, 0.5},
		{1, 0.7310585786300049},
		{2, 0.8807970779778823},
		{3, 0.9525741268224331},
	} {
		if got := sigmoid(tc.input); !nearlyEqual(got, tc.want, eps) {
			t.Errorf("Failed test No.%v: got = %v, want = %v", num, got, tc.want)
		}
	}
}

const eps = 1e-5

func nearlyEqual(a, b, epsilon float64) bool {
	return math.Abs(a-b) < epsilon
}

func TestDot(t *testing.T) {
	type Input struct {
		a, b []float64
	}
	for num, tc := range []struct {
		input Input
		want  float64
	}{
		{Input{a: []float64{1, 2}, b: []float64{2, 1}}, 4},
		{Input{a: []float64{0, 0}, b: []float64{0, 0}}, 0},
		{Input{a: []float64{1, 1}, b: []float64{1, 1}}, 2},
		{Input{a: []float64{-1, -1}, b: []float64{1, 1}}, -2},
		{Input{a: []float64{3, 4}, b: []float64{5, 6}}, 39},
		{Input{a: []float64{1.5, 2.5}, b: []float64{2.5, 1.5}}, 7.5},
		{Input{a: []float64{0.5, 0.5}, b: []float64{0.5, 0.5}}, 0.5},
	} {
		if got := dot(tc.input.a, tc.input.b); !nearlyEqual(got, tc.want, eps) {
			t.Errorf("Failed test No.%v: got = %v, want = %v", num, got, tc.want)
		}
	}
}
