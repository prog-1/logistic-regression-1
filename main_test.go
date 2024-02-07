package main

import (
	"math"
	"testing"
)

const eps = 1e-5

func nearlyEqual(a, b, epsilon float64) bool {
	return math.Abs(a-b) < epsilon
}

func TestDot(t *testing.T) {
	type Input struct {
		a, b []float64
	}
	for _, tc := range []struct {
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
			t.Errorf("dot(%v, %v) = %v, want = %v", tc.input.a, tc.input.b, got, tc.want)
		}
	}
}

func TestSigmoid(t *testing.T) {
	for _, tc := range []struct {
		input, want float64
	}{
		{-1, 0.2689414213699951},
		{0, 0.5},
		{1, 0.7310585786300049},
		{2, 0.8807970779778823},
		{3, 0.9525741268224331},
	} {
		if got := sigmoid(tc.input); !nearlyEqual(got, tc.want, eps) {
			t.Errorf("sigmoid(%v) = %v, want = %v", tc.input, got, tc.want)
		}
	}
}

func TestInference(t *testing.T) {
	type Input struct {
		inputs [][]float64
		w      []float64
		b      float64
	}
	for _, tc := range []struct {
		input Input
		want  []float64
	}{
		{Input{inputs: [][]float64{{50, 50}}, w: []float64{0.5, 0.5}, b: -50}, []float64{0.5}},          // Point is on the hyperplane
		{Input{inputs: [][]float64{{50, 50}, {1, -29}}, w: []float64{0, 0}, b: 0}, []float64{0.5, 0.5}}, // All weights are 0
	} {
		got := inference(tc.input.inputs, tc.input.w, tc.input.b)
		for i := range got {
			if !nearlyEqual(got[i], tc.want[i], eps) {
				t.Errorf("inference(%v) = %v, want = %v", tc.input, got, tc.want)
			}
		}
	}
}
