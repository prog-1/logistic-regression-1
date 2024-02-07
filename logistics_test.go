package main

import (
	"math"
	"testing"
)

const (
	e = 0.1 //range (error) of answers
)

func TestSigmoid(t *testing.T) {
	for _, tc := range []struct {
		name  string
		input float64
		want  float64
	}{
		{"1", -1, 0.26},
		{"2", 0, 0.5},
		{"3", 1, 0.73},
		{"4", 2, 0.88},
		{"5", 3, 0.95},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := sigmoid(tc.input)
			if math.Abs(got-tc.want) > e {
				t.Errorf("got = %v, want = %v", got, tc.want)
			}
		})
	}
}

func TestDot(t *testing.T) {
	for _, tc := range []struct {
		name string
		a    []float64
		b    []float64
		want float64
	}{
		{"1", []float64{1, 2}, []float64{3, 4}, 11},
		{"2", []float64{0, 0}, []float64{0, 0}, 0},
		{"3", []float64{2, 2}, []float64{0, 0}, 0},
		{"4", []float64{-2, -100}, []float64{-200, 300}, -29600},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := dot(tc.a, tc.b)
			if math.Abs(got-tc.want) > e {
				t.Errorf("got = %v, want = %v", got, tc.want)
			}
		})
	}
}

func TestInference(t *testing.T) {
	for _, tc := range []struct {
		name   string
		inputs [][]float64
		w      []float64
		b      float64
		want   []float64
	}{
		{
			name:   "1",
			inputs: [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			w:      []float64{0.1, 0.2, 0.3},
			b:      -0.5,
			want:   []float64{0.71, 0.94, 0.99},
		},
		{
			name:   "2",
			inputs: [][]float64{},
			w:      []float64{0.1, 0.2, 0.3},
			b:      -0.5,
			want:   []float64{},
		},
		{
			name:   "3",
			inputs: [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			w:      []float64{0, 0, 0},
			b:      0,
			want:   []float64{0.5, 0.5, 0.5},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := inference(tc.inputs, tc.w, tc.b)
			for i := range got {
				if math.Abs(got[i]-tc.want[i]) > e {
					t.Errorf("got = %v, want = %v", got, tc.want)
				}
			}

		})
	}
}

func main() {
	testing.Main(
		/* matchString */ func(a, b string) (bool, error) { return a == b, nil },
		/* tests */ []testing.InternalTest{
			{Name: "Test Sigmoid", F: TestSigmoid},
			{Name: "Test Dot", F: TestDot},
			{Name: "Test Inference", F: TestInference},
		},
		/* benchmarks */ nil /* examples */, nil)
}
