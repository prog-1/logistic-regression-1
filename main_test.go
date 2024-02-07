package main

import "testing"

func Abs(a float64) float64 {
	if a < 0 {
		return -a
	}
	return a
}

func isEqual(a, b float64, e float64) bool {
	return Abs(a-b) < e
}

func TestSigmoid(t *testing.T) {
	for _, tc := range []struct {
		input  float64
		output float64
	}{
		{-1, 0.2689414213699951207488407581781637256348553598349434807236340920809595469297953606125254679240187547078255335068592254335588672},
		{0, 0.5},
		{1, 0.7310585786300048792511592418218362743651446401650565192763659079190404530702046393874745320759812452921744664931407745664411327},
		{2, 0.8807970779778824440597291413023967952063842986289682757984052500609766222883192417294737608368383572109513798507770387661841545},
		{3, 0.9525741268224332191211518482282477986138205675793908992821119912255512884972897661142163455089399609940819230332495665531130178},
	} {
		if res := sigmoid(tc.input); !isEqual(res, tc.output, 1e-15) {
			t.Errorf("sigmoid(%v) = %v, expected - %v", tc.input, res, tc.output)
		}
	}
}

func TestDot(t *testing.T) {
	for _, tc := range []struct {
		a, b []float64
		exp  float64
	}{
		{[]float64{1, 2, 3}, []float64{0.1, 0.2, 0.3}, 1.4},
		{[]float64{1, -2, 3}, []float64{0.1, 0.2, 0.3}, 0.6},
		{[]float64{-1, 2, -3}, []float64{0.1, 0.2, 0.3}, -0.6},
	} {
		if res := dot(tc.a, tc.b); !isEqual(res, tc.exp, 1e-10) {
			t.Errorf("dot(%v, %v) = %v, expected - %v", tc.a, tc.b, res, tc.exp)

		}
	}
}

func TestInfrence(t *testing.T) {
	tc := struct {
		inputs  [][]float64
		weights []float64
		b       float64
		want    []float64
	}{
		[][]float64{
			[]float64{1, 2, 3},
			[]float64{1, -2, 3},
			[]float64{-1, 2, -3},
		},
		[]float64{0.1, 0.2, 0.3},
		0,
		[]float64{0.8021838885585817481543435915519132375077833237084304443062099967, 0.6456563062257954529091106364118829640767984024588060872071562097, 0.3543436937742045470908893635881170359232015975411939127928437902},
	}
	res := inference(tc.inputs, tc.weights, tc.b)
	for i := range tc.inputs {
		if !isEqual(res[i], tc.want[i], 1e-10) {
			t.Errorf("inference(%v, %v, %v) = %v, expected - %v", tc.inputs[i], tc.weights, tc.b, res[i], tc.want[i])

		}
	}
}
