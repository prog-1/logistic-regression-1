package main

import (
	"math"
	"testing"
)

func TestSigmoid(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{0, 0.5},
		{-1, 0.2689414213699951},
		{1, 0.7310585786300049},
		{2, 0.8807970779778823},
		{3, 0.9525741268224334},
		{100, 1},
		{-100, 0},
		{math.Inf(1), 1},
	}

	for _, test := range tests {
		output := sigmoid(test.input)
		if math.Abs(output-test.expected) > 1e-9 {
			t.Errorf("Expected sigmoid(%f) to be %f, but got %f", test.input, test.expected, output)
		}
	}
}

func TestDot(t *testing.T) {
	tests := []struct {
		a        []float64
		b        []float64
		expected float64
	}{
		{[]float64{1, 2, 3}, []float64{4, 5, 6}, 32},
		{[]float64{0, 0, 0}, []float64{1, 2, 3}, 0},
		{[]float64{-1, -2, -3}, []float64{1, 2, 3}, -14},
		{[]float64{1.5, 2.5, 3.5}, []float64{2, 3, 4}, 24.5},
	}

	for _, test := range tests {
		output := dot(test.a, test.b)
		if output != test.expected {
			t.Errorf("Expected dot(%v, %v) to be %f, but got %f", test.a, test.b, test.expected, output)
		}
	}
}
func TestInference(t *testing.T) {
	tests := []struct {
		inputs   [][]float64
		weights  []float64
		bias     float64
		expected []float64
	}{
		{
			inputs: [][]float64{
				{1, 2, 3},
				{4, 5, 6},
			},
			weights:  []float64{0.5, 0.5, 0.5},
			bias:     1,
			expected: []float64{0.9241418199787566, 0.9933071490757153},
		},
		{
			inputs: [][]float64{
				{0, 0, 0},
				{1, 1, 1},
			},
			weights:  []float64{0.1, 0.2, 0.3},
			bias:     -0.5,
			expected: []float64{0.3775406687981454, 0.6224593312018546},
		},
	}

	for _, test := range tests {
		outputs := inference(test.inputs, test.weights, test.bias)
		if !floatSlicesEqual(outputs, test.expected) {
			t.Errorf("Expected inference(%v, %v, %f) to be %v, but got %v", test.inputs, test.weights, test.bias, test.expected, outputs)
		}
	}
}

func floatSlicesEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1 {
			return false
		}
	}
	return true
}
func TestDCost(t *testing.T) {
	tests := []struct {
		inputs   [][]float64
		y        []float64
		p        []float64
		expected struct {
			dw []float64
			db float64
		}
	}{
		{
			inputs: [][]float64{
				{1, 2, 3},
				{4, 5, 6},
			},
			y: []float64{0, 1},
			p: []float64{1, 0},
			expected: struct {
				dw []float64
				db float64
			}{
				dw: []float64{-1.5, -1.5, -1.5},
				db: 0,
			},
		},
	}

	for _, test := range tests {
		dw, db := dCost(test.inputs, test.y, test.p)
		if !floatSlicesEqual(dw, test.expected.dw) {
			t.Errorf("Expected dCost(%v, %v, %v) to return dw %v, but got %v", test.inputs, test.y, test.p, test.expected.dw, dw)
		}
		if db != test.expected.db {
			t.Errorf("Expected dCost(%v, %v, %v) to return db %f, but got %f", test.inputs, test.y, test.p, test.expected.db, db)
		}
	}
}
