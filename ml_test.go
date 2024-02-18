package main

import (
	"math"
	"reflect"
	"testing"
)

const eps = 1e-5

func nearlyEqualFloat(a, b, epsilon float64) bool {
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
		if got := dot(tc.input.a, tc.input.b); !nearlyEqualFloat(got, tc.want, eps) {
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
		if got := sigmoid(tc.input); !nearlyEqualFloat(got, tc.want, eps) {
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
		// {Input{inputs: [][]float64{{1,1}{}}}, 0.5},// Hyperspace is parallel to the axis
		// i.e. points are distributed chaotically, so the only way to split
		// // one type through another is via utilization of extra dimension.
	} {
		got := inference(tc.input.inputs, tc.input.w, tc.input.b)
		for i := range got {
			if !nearlyEqualFloat(got[i], tc.want[i], eps) {
				t.Errorf("inference(%v) = %v, want = %v", tc.input, got, tc.want)
			}
		}
	}
}

func TestDCost(t *testing.T) {
	type Input struct {
		inputs [][]float64
		y, p   []float64
	}
	type Output struct {
		dw []float64
		db float64
	}
	nearlyEqualOutput := func(got, want Output, epsilon float64) bool {
		if !nearlyEqualFloat(got.db, want.db, epsilon) || len(got.dw) != len(want.dw) {
			return false
		}
		for j := range got.dw {
			if !nearlyEqualFloat(got.dw[j], want.dw[j], epsilon) {
				return false
			}
		}
		return true
	}
	for _, tc := range []struct {
		input Input
		want  Output
	}{
		{input: Input{[][]float64{}, []float64{}, []float64{}}, want: Output{[]float64{}, 0}},                           // No input
		{input: Input{[][]float64{{1, 2}, {2, 7}}, []float64{3, 9}, []float64{3, 9}}, want: Output{[]float64{0, 0}, 0}}, // Complete match
	} {
		var got Output
		got.dw, got.db = dCost(tc.input.inputs, tc.input.y, tc.input.p)
		if !nearlyEqualOutput(got, tc.want, eps) {
			t.Errorf("dCost(%v) = %v, want = %v", tc.input, got, tc.want)
		}
	}
}

func TestSplit(t *testing.T) {
	type Input struct {
		inputs [][]float64
		y      []float64
		ratio  int
	}
	type Output struct {
		xTrain, xTest [][]float64
		yTrain, yTest []float64
	}
	for _, tc := range []struct {
		input Input
		want  Output
	}{
		{input: Input{}, want: Output{}},
		{input: Input{inputs: [][]float64{{1, 10}, {10, 1}}, y: []float64{0, 1}, ratio: 50},
			want: Output{xTrain: [][]float64{{10, 1}}, xTest: [][]float64{{1, 10}}, yTrain: []float64{1}, yTest: []float64{0}}},
		{input: Input{inputs: [][]float64{{1, 10}, {10, 1}}, y: []float64{0, 1}, ratio: 0},
			want: Output{xTrain: [][]float64{{1, 10}, {10, 1}}, xTest: [][]float64{}, yTrain: []float64{0, 1}, yTest: []float64{}}},
		{input: Input{inputs: [][]float64{{1, 10}, {10, 1}}, y: []float64{0, 1}, ratio: 100},
			want: Output{xTrain: [][]float64{}, xTest: [][]float64{{1, 10}, {10, 1}}, yTrain: []float64{}, yTest: []float64{0, 1}}},
		{input: Input{inputs: [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}}, y: []float64{1, 0, 1, 0, 0}, ratio: 20}, // (c) Sebastjans Peive
			want: Output{xTrain: [][]float64{{3, 4}, {5, 6}, {7, 8}, {9, 10}}, xTest: [][]float64{{1, 2}}, yTrain: []float64{0, 1, 0, 0}, yTest: []float64{1}}},
	} {
		var got Output
		got.xTrain, got.xTest, got.yTrain, got.yTest = split(tc.input.inputs, tc.input.y, tc.input.ratio)
		if !reflect.DeepEqual(tc.want.xTrain, got.xTrain) {
			t.Errorf("got.xTrain = %v, want.xTrain %v", got.xTrain, tc.want.xTrain)
		} else if !reflect.DeepEqual(tc.want.xTest, got.xTest) {
			t.Errorf("got.xTest = %v, want.xTest = %v", got.xTest, tc.want.xTest)
		} else if !reflect.DeepEqual(tc.want.yTrain, got.yTrain) {
			t.Errorf("got.yTrain = %v, want.yTrain %v", got.yTrain, tc.want.yTrain)
		} else if !reflect.DeepEqual(tc.want.yTest, got.yTest) {
			t.Errorf("got.yTest = %v, want.yTest %v", got.yTest, tc.want.yTest)
		}
	}
}

func TestAccuracy(t *testing.T) {
	type Input struct {
		inputs [][]float64
		y      []float64
		w      []float64
		b      float64
	}
	for _, tc := range []struct {
		input Input
		want  float64
	}{
		{Input{inputs: [][]float64{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}}, y: []float64{1, 1, 1, 1, 1}, w: []float64{1, 1}, b: 1}, 1},
		{Input{inputs: [][]float64{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}}, y: []float64{0, 0, 0, 0, 0}, w: []float64{1, 1}, b: 1}, 0},
		{Input{inputs: [][]float64{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}}, y: []float64{1, 1, 1, 0, 0, 0}, w: []float64{1, 1}, b: 1}, 0.5},
	} {
		if got := accuracy(tc.input.inputs, tc.input.y, tc.input.w, tc.input.b); !nearlyEqualFloat(got, tc.want, eps) {
			t.Errorf("accuracy(%v) = %v, want = %v", tc.input, got, tc.want)
		}
	}
}
