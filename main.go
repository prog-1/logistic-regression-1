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

func main() {
	testing.Main(
		func(a, b string) (bool, error) { return a == b, nil },
		[]testing.InternalTest{
			{Name: "Test SecondLargest", F: TestSigmoid},
		}, nil, nil,
	)
}
