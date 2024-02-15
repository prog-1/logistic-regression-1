package main

import (
	"math"
	"reflect"
	"testing"
)

const eps = 1e-5

func TestSigmoid(t *testing.T) {
	for _, tc := range []struct {
		name string
		z    float64
		want float64
	}{
		{"case-1", -1, 0.268941},
		{"case-2", 0, 0.5},
		{"case-3", 1, 0.731059},
		{"case-4", 2, 0.880797},
		{"case-5", 3, 0.952574},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := sigmoid(tc.z); math.Abs(got-tc.want) > eps {
				t.Errorf("got = %v, want = %v", got, tc.want)
			}
		})
	}
}

func TestDot(t *testing.T) {
	for _, tc := range []struct {
		name string
		a, b []float64
		want float64
	}{
		{"case-1", []float64{0, 0}, []float64{0, 1}, 0},
		{"case-2", []float64{0, 100}, []float64{1, 1}, 100},
		{"case-3", []float64{100, 0}, []float64{2, 1}, 200},
		{"case-4", []float64{50, 50}, []float64{0.5, 3}, 175},
		{"case-5", []float64{-25, -75}, []float64{1.75, 2.1}, -201.25},
		{"case-6", []float64{10.5, -60}, []float64{2, 0.3}, 3},
		{"case-7", []float64{-25.5, 55.5}, []float64{1.4, 0.15}, -27.375},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := dot(tc.a, tc.b); math.Abs(got-tc.want) > eps {
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
		{"case-1", [][]float64{{50, 50}}, []float64{0.2, 0.2}, 10, []float64{1}},
		{"case-2", [][]float64{{100, 0}}, []float64{0.3, 1}, -25, []float64{0.993307}},
		{"case-3", [][]float64{{0, 100}, {50, 50}}, []float64{0.2, 0.15}, -12.5, []float64{0.924142, 0.993307}},
		{"case-4", [][]float64{{-5, -15}, {10.5, -30}, {-25.5, 25.5}}, []float64{1.2, 0.75}, 11.1, []float64{0.00212894, 0.768525, 0.407333}},
		{"case-5", [][]float64{{10, 2}, {50, 25}}, []float64{0, 0}, 0, []float64{0.5, 0.5}},
		{"case-6", [][]float64{{10, 2}, {50, 5}}, []float64{0, 0.5}, 1, []float64{0.880797, 0.970688}},
		{"case-7", [][]float64{{20, 10}}, []float64{1, 2}, -40, []float64{0.5}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := inference(tc.inputs, tc.w, tc.b)
			for i := range got {
				if math.Abs(got[i]-tc.want[i]) > eps {
					t.Errorf("got[%v] = %v, want[%v] = %v", i, got[i], i, tc.want[i])
				}
			}
		})
	}
}

func TestDCost(t *testing.T) {
	for _, tc := range []struct {
		name         string
		inputs       [][]float64
		y, p, wantDw []float64
		wantDb       float64
	}{
		{"case-1", [][]float64{{50, 50}}, []float64{1}, []float64{1}, []float64{0, 0}, 0},
		{"case-2", [][]float64{{100, 0}}, []float64{1}, []float64{0}, []float64{-100, 0}, -1},
		{"case-3", [][]float64{{50, 5}, {80, 25}}, []float64{0, 0}, []float64{0, 1}, []float64{40, 12.5}, 0.5},
		{"case-4", [][]float64{{-5, -15}, {10.5, -30}, {55.5, 45.5}}, []float64{0, 0, 1}, []float64{0, 0, 0}, []float64{-18.5, -15.166666}, -0.333333},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotDw, gotDb := dCost(tc.inputs, tc.y, tc.p)
			for i := range gotDw {
				if math.Abs(gotDw[i]-tc.wantDw[i]) > eps {
					t.Errorf("gotDw[%v] = %v, wantDw[%v] = %v", i, gotDw[i], i, tc.wantDw[i])
				}
			}
			if math.Abs(gotDb-tc.wantDb) > eps {
				t.Errorf("gotDb = %v, wantDb = %v", gotDb, tc.wantDb)
			}
		})
	}
}

func TestSplit(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		inputs                [][]float64
		y                     []float64
		wantXTrain, wantXTest [][]float64
		wantYTrain, wantYTest []float64
	}{
		{"case-1", [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}}, []float64{1, 0, 1, 0, 0},
			[][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}}, [][]float64{{9, 10}}, []float64{1, 0, 1, 0}, []float64{0}},
		{"case-2", [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}, {13, 14}, {15, 16}, {17, 18}, {19, 20}}, []float64{1, 0, 1, 0, 0, 1, 1, 1, 0, 1},
			[][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}, {13, 14}, {15, 16}}, [][]float64{{17, 18}, {19, 20}}, []float64{1, 0, 1, 0, 0, 1, 1, 1}, []float64{0, 1}},
		{"case-3", [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}}, []float64{1, 0, 1, 0, 0, 1},
			[][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}}, [][]float64{{9, 10}, {11, 12}}, []float64{1, 0, 1, 0}, []float64{0, 1}},
		{"case-4", [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}, {13, 14}, {15, 16}, {17, 18}, {19, 20}, {21, 22}}, []float64{1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0},
			[][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}, {13, 14}, {15, 16}}, [][]float64{{17, 18}, {19, 20}, {21, 22}}, []float64{1, 0, 1, 0, 0, 1, 1, 1}, []float64{0, 1, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotXTrain, gotXTest, gotYTrain, gotYTest := split(tc.inputs, tc.y)
			if !reflect.DeepEqual(gotXTrain, tc.wantXTrain) {
				t.Errorf("gotXTrain = %v, wantXTrain = %v", gotXTrain, tc.wantXTrain)
			}
			if !reflect.DeepEqual(gotXTest, tc.wantXTest) {
				t.Errorf("gotXTest = %v, wantXTest = %v", gotXTest, tc.wantXTest)
			}
			if !reflect.DeepEqual(gotYTrain, tc.wantYTrain) {
				t.Errorf("gotYTrain = %v, wantYTrain = %v", gotYTrain, tc.wantYTrain)
			}
			if !reflect.DeepEqual(gotYTest, tc.wantYTest) {
				t.Errorf("gotYTest = %v, wantYTest = %v", gotYTest, tc.wantYTest)
			}
		})
	}
}

func TestAccuracy(t *testing.T) {
	for _, tc := range []struct {
		name   string
		inputs [][]float64
		y, w   []float64
		b      float64
		want   float64
	}{
		{"case-1", [][]float64{{100, 0}}, []float64{1}, []float64{0.3, 1}, -25, 100},
		{"case-2", [][]float64{{0, 100}, {50, 50}}, []float64{1, 1}, []float64{0.2, 0.15}, -12.5, 100},
		{"case-3", [][]float64{{0, 100}, {50, 50}}, []float64{1, 0}, []float64{0.2, 0.15}, -12.5, 50},
		{"case-4", [][]float64{{0, 100}, {50, 50}}, []float64{0, 0}, []float64{0.2, 0.15}, -12.5, 0},
		{"case-5", [][]float64{{-5, -15}, {10.5, -30}, {-25.5, 25.5}}, []float64{0, 1, 0}, []float64{1.2, 0.75}, 11.1, 100},
		{"case-6", [][]float64{{-5, -15}, {10.5, -30}, {-25.5, 25.5}}, []float64{0, 0, 0}, []float64{1.2, 0.75}, 11.1, 66.666666},
		{"case-7", [][]float64{{-5, -15}, {10.5, -30}, {-25.5, 25.5}}, []float64{1, 1, 1}, []float64{1.2, 0.75}, 11.1, 33.333333},
		{"case-8", [][]float64{{-5, -15}, {10.5, -30}, {-25.5, 25.5}}, []float64{1, 0, 1}, []float64{1.2, 0.75}, 11.1, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := accuracy(tc.inputs, tc.y, tc.w, tc.b); math.Abs(got-tc.want) > eps {
				t.Errorf("got = %v, want = %v", got, tc.want)
			}
		})
	}
}
