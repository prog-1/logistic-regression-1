package main

import (
	"math"
	"reflect"
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
		{inputs: [][]float64{
			{0, -1}, {0, 0}, {0, 1},
			{-1, 1}, {-1, 0}, {-1, -1},
			{1, 1}, {1, 0}, {1, -1}}, w: []float64{1, 0}, b: 0, want: []float64{
			0.5, 0.5, 0.5,
			0.268941, 0.268941, 0.268941,
			0.731059, 0.731059, 0.731059}},
		// *   *   *
		//     |
		// *   *   *
		//     |
		// *   *   *
	} {
		got := inference(tc.inputs, tc.w, tc.b)
		for i := range got {
			if math.Abs(got[i]-tc.want[i]) > 1e-6 {
				t.Errorf("got = %v, want = %v", got, tc.want)
			}
		}
	}
}

func TestDCost(t *testing.T) {
	for _, tc := range []struct {
		inputs      [][]float64
		y, p, want1 []float64
		want2       float64
	}{
		{inputs: [][]float64{{10, 20}, {30, -50}, {5, 5}}, y: []float64{1, 2, 3}, p: []float64{3, 2, 5}, want1: []float64{10, 16.666667}, want2: 1.333333},
		{inputs: [][]float64{{10, 20}, {30, -50}}, y: []float64{2, 1}, p: []float64{2, 1}, want1: []float64{0, 0}, want2: 0},
		{inputs: [][]float64{{1, 2}, {3, 1}}, y: []float64{1, 1}, p: []float64{2, 2}, want1: []float64{2, 1.5}, want2: 1},
	} {
		got1, got2 := dCost(tc.inputs, tc.y, tc.p)
		for i := range got1 {
			if math.Abs(got1[i]-tc.want1[i]) > 1e-6 {
				t.Errorf("got1 = %v, want1 = %v", got1, tc.want1)
			}
		}
		if math.Abs(got2-tc.want2) > 1e-6 {
			t.Errorf("got2 = %v, want2 = %v", got2, tc.want2)
		}
	}
}

func TestAccuracy(t *testing.T) {
	for _, tc := range []struct {
		inputs  [][]float64
		y, w    []float64
		b, want float64
	}{
		{inputs: [][]float64{{10, 20}, {30, -50}, {5, 5}}, y: []float64{0, 1, 1}, w: []float64{0.1, 0.05}, b: 0.25, want: 0.666667},
		{inputs: [][]float64{{10, 20}, {30, -50}, {5, 5}}, y: []float64{1, 1, 0}, w: []float64{0.1, 0.05}, b: 0.25, want: 0.666667},
		{inputs: [][]float64{{10, 20}, {30, -50}, {5, 5}}, y: []float64{1, 1, 1}, w: []float64{0.1, 0.05}, b: 0.25, want: 1},
		{inputs: [][]float64{{10, 20}, {30, -50}, {5, 5}}, y: []float64{0, 0, 1}, w: []float64{0.1, 0.05}, b: 0.25, want: 0.333333},
		{inputs: [][]float64{{10, 20}, {30, -50}, {5, 5}}, y: []float64{1, 0, 0}, w: []float64{0.1, 0.05}, b: 0.25, want: 0.333333},
		{inputs: [][]float64{{10, 20}, {30, -50}, {5, 5}}, y: []float64{0, 1, 0}, w: []float64{0.1, 0.05}, b: 0.25, want: 0.333333},
		{inputs: [][]float64{{10, 20}, {30, -50}, {5, 5}}, y: []float64{0, 0, 0}, w: []float64{0.1, 0.05}, b: 0.25, want: 0},
		{inputs: [][]float64{{10, 40, 25}, {0, 100, -300}}, y: []float64{1, 1}, w: []float64{0.05, 0.05, 0.02}, b: 0, want: 0.5},
		{inputs: [][]float64{
			{0, -1}, {0, 0}, {0, 1},
			{-1, 1}, {-1, 0}, {-1, -1},
			{1, 1}, {1, 0}, {1, -1}}, y: []float64{1, 1, 1, 0, 0, 1, 1, 1, 1}, w: []float64{1, 0}, b: 0, want: 0.888889},
	} {
		got := accuracy(tc.inputs, tc.y, tc.w, tc.b)
		if math.Abs(got-tc.want) > 1e-6 {
			t.Errorf("got = %v, want = %v", got, tc.want)
		}
	}
}

func TestSplit(t *testing.T) {
	for _, tc := range []struct {
		inputs, wantXT, wantXTest [][]float64
		y, wantYT, wantYTest      []float64
	}{
		{inputs: [][]float64{{10, 13}, {3, 20}, {12, 15}, {30, 20}, {10, 10}, {3, 9}, {1, 1}, {2, 2}},
			y:      []float64{1, 1, 1, 0, 1, 1, 1, 1},
			wantXT: [][]float64{{3, 20}, {12, 15}, {30, 20}, {10, 10}, {3, 9}, {1, 1}, {2, 2}}, wantXTest: [][]float64{{10, 13}},
			wantYT: []float64{1, 1, 0, 1, 1, 1, 1}, wantYTest: []float64{1}},

		{inputs: [][]float64{{1, 5}, {2, 4}, {4, 10}, {20, 30}, {25, 43}, {40, 50}, {60, 60}, {65, 55}, {70, 80}, {90, 100}},
			y:         []float64{1, 1, 1, 0, 1, 0, 1, 1, 0, 0},
			wantXT:    [][]float64{{4, 10}, {20, 30}, {25, 43}, {40, 50}, {60, 60}, {65, 55}, {70, 80}, {90, 100}},
			wantXTest: [][]float64{{1, 5}, {2, 4}},
			wantYT:    []float64{1, 0, 1, 0, 1, 1, 0, 0},
			wantYTest: []float64{1, 1}},

		{inputs: [][]float64{{1, 5}, {2, 4}, {3, 10}, {20, 50}, {30, 60}},
			y:         []float64{1, 0, 0, 0, 0},
			wantXT:    [][]float64{{2, 4}, {3, 10}, {20, 50}, {30, 60}},
			wantXTest: [][]float64{{1, 5}},
			wantYT:    []float64{0, 0, 0, 0},
			wantYTest: []float64{1}},
	} {
		gotXT, gotXTest, gotYT, gotYTest := split(tc.inputs, tc.y)
		if !reflect.DeepEqual(gotXT, tc.wantXT) {
			t.Errorf("gotXT = %v, wantXT = %v", gotXT, tc.wantXT)
		}
		if !reflect.DeepEqual(gotXTest, tc.wantXTest) {
			t.Errorf("gotXTest = %v, wantXTest = %v", gotXTest, tc.wantXTest)
		}
		if !reflect.DeepEqual(gotYT, tc.wantYT) {
			t.Errorf("gotYT = %v, wantYT = %v", gotYT, tc.wantYT)
		}
		if !reflect.DeepEqual(gotYTest, tc.wantYTest) {
			t.Errorf("gotYTest = %v, wantYTest = %v", gotYTest, tc.wantYTest)
		}
	}
}
