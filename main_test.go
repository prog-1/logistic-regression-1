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
		{z: -1.0, want: 0.268941},
		{z: 0.0, want: 0.5},
		{z: 1.0, want: 0.731059},
		{z: 2.0, want: 0.880797},
		{z: 3.0, want: 0.952574},
	} {
		got := sigmoid(tc.z)
		if math.Abs(got-tc.want) > 1e-6 {
			t.Errorf("got = %v, want = %v", got, tc.want)
		}

	}
}

func TestDot(t *testing.T) {
	for _, tc := range []struct {
		a, b []float64
		want float64
	}{
		{a: []float64{}, b: []float64{}, want: 0},
		{a: []float64{}, b: []float64{1, 2, 3}, want: 0},
		{a: []float64{1, 2}, b: []float64{1, 2, 3}, want: 0},
		{a: []float64{10, 20}, b: []float64{0.1, 0.05}, want: 2},
		{a: []float64{30, -50}, b: []float64{0.1, 0.05}, want: 0.5},
		{a: []float64{5, 5}, b: []float64{0.1, 0.05}, want: 0.75},
		{a: []float64{10, 40, 25}, b: []float64{0.05, 0.05, 0.02}, want: 3},
	} {
		got := dot(tc.a, tc.b)
		if got != tc.want {
			t.Errorf("got = %v, want = %v", got, tc.want)
		}

	}
}

func TestInference(t *testing.T) {
	for _, tc := range []struct {
		inputs [][]float64
		w      []float64
		b      float64
		want   []float64
	}{
		{inputs: [][]float64{{10, 20}, {30, -50}, {5, 5}}, w: []float64{0.1, 0.05}, b: 0.25, want: []float64{0.904651, 0.679179, 0.731059}},
		{inputs: [][]float64{{10, 40, 25}, {0, 100, -300}}, w: []float64{0.05, 0.05, 0.02}, b: 0, want: []float64{0.952574, 0.268941}},
		{inputs: [][]float64{{0.4, 0.6}, {0.2, 0.8}, {0.5, 0.5}}, w: []float64{1, 1}, b: -1, want: []float64{0.5, 0.5, 0.5}},
		{inputs: [][]float64{{0.4, 0.6}, {0.6, 0.8}, {-0.2, 0.8}}, w: []float64{1, 1}, b: -1, want: []float64{0.5, 0.598688, 0.401312}},
		{inputs: [][]float64{{0.4, 0.6}, {0.6, 0.8}, {-0.2, 0.8}}, w: []float64{0, 0}, b: -1, want: []float64{0.268941, 0.268941, 0.268941}},
		{inputs: [][]float64{{0.4, 0.6}, {0.6, 0.8}, {-0.2, 0.8}}, w: []float64{0, 0}, b: 0, want: []float64{0.5, 0.5, 0.5}},
		{inputs: [][]float64{{3, 1, 2}, {3, -2, 5}, {3, 0, 0}}, w: []float64{1, 0, 0}, b: 3, want: []float64{0.997527, 0.997527, 0.997527}},
	} {
		got := inference(tc.inputs, tc.w, tc.b)
		for i := range got {
			if math.Abs(got[i]-tc.want[i]) > 1e-6 {
				t.Errorf("got = %v, want = %v", got, tc.want)
			}
		}
	}
}
