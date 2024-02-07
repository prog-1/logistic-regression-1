package main

import (
	"math"
	"testing"
)

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Pow(math.E, -z))
}

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
		if got := sigmoid(tc.input); got != tc.want {
			t.Errorf("Failed test No.%v: got = %v, want = %v", num, got, tc.want)
		}
	}
}

func dot(a []float64, b []float64) (res float64) {
	for i := range a {
		res += a[i] * b[i]
	}
	return res
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
		if got := dot(tc.input.a, tc.input.b); got != tc.want {
			t.Errorf("Failed test No.%v: got = %v, want = %v", num, got, tc.want)
		}
	}
}

func prediction(x []float64, w []float64, b float64) float64 {
	return sigmoid(dot(w, x) + b)
}

func inference(inputs [][]float64, w []float64, b float64) (probabilities []float64) {
	for _, x := range inputs {
		probabilities = append(probabilities, prediction(x, w, b))
	}
	return probabilities
}

func main() {
	testing.Main(
		func(a, b string) (bool, error) { return a == b, nil },
		[]testing.InternalTest{
			{Name: "Test Sigmoid", F: TestSigmoid},
			{Name: "Test Dot", F: TestDot},
		}, nil, nil,
	)
}
