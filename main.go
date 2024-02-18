package main

import (
	"encoding/csv"
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette/moreland"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}
func dot(a []float64, b []float64) (dot float64) {
	if len(a) != len(b) {
		fmt.Println(a, b)
		panic("len(a) != len(b)")
	}
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
	}
	return dot
}

func inference(inputs [][]float64, w []float64, b float64) (res []float64) {
	res = make([]float64, len(inputs))
	for i := 0; i < len(inputs); i++ {
		res[i] = sigmoid(dot(inputs[i], w) + b)
	}
	return res
}
func dCost(inputs [][]float64, y, p []float64) (dw []float64, db float64) {
	dw = make([]float64, len(inputs[0]))
	m := len(inputs)
	for i := 0; i < m; i++ {
		for j := 0; j < len(inputs[0]); j++ {
			dw[j] += (inputs[i][j] * (p[i] - y[i])) / float64(m)
		}
		db += (p[i] - y[i]) / float64(m)
	}
	return dw, db
}

func gradientDescent(inputs [][]float64, y, w []float64, alpha, b float64, epochs int) ([]float64, float64, []float64, float64) {
	var dw []float64
	var db float64
	for i := 0; i < epochs; i++ {
		p := inference(inputs, w, b)
		dw, db = dCost(inputs, y, p)
		for j := 0; j < len(w); j++ {
			w[j] -= alpha * dw[j]
		}
		b -= alpha * db
		//fmt.Println(dw, db)
		//fmt.Println(w, b)
	}
	return w, b, dw, db
}

func accuracy(inputs [][]float64, y []float64, w []float64, b float64) float64 {
	prediction := inference(inputs, w, b)
	var truePos, trueNeg, falsePos, falseNeg float64
	for i := 0; i < len(y); i++ {
		if prediction[i] >= 0.5 {
			if y[i] == 1 {
				truePos++
			} else {
				falsePos++
			}
		} else {
			if y[i] == 0 {
				trueNeg++
			} else {
				falseNeg++
			}
		}
	}
	return (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
}

func split(data [][]string) (xTrain, xTest [][]float64, yTrain, yTest []float64) {
	half := len(data) / 2
	segment := len(data[0])
	xTrain = make([][]float64, half)
	for i := range xTrain {
		xTrain[i] = make([]float64, segment-1)
	}
	yTrain = make([]float64, half)

	xTest = make([][]float64, half)
	for i := range xTest {
		xTest[i] = make([]float64, segment-1)
	}
	yTest = make([]float64, half)

	for i, row := range data[:half] {
		for j := 0; j < 2; j++ {
			xTrain[i][j], _ = strconv.ParseFloat(row[j], 64)
		}
		yTrain[i], _ = strconv.ParseFloat(row[2], 64)
	}

	for i, row := range data[half:] {
		for j := 0; j < 2; j++ {
			xTest[i][j], _ = strconv.ParseFloat(row[j], 64)
		}
		yTest[i], _ = strconv.ParseFloat(row[2], 64)
	}
	//fmt.Println(xTrain, xTest, yTrain, yTest)
	return xTrain, xTest, yTrain, yTest
}

type plottable struct {
	grid       [][]float64
	N          int
	M          int
	resolution float64
	minX       float64
	minY       float64
}

func main() {
	//reading
	file, err := os.Open("data/exams1.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	data, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	// variables
	p := plot.New()
	var dw []float64
	var db float64
	xTrain, xTest, yTrain, yTest := split(data)
	w := make([]float64, len(xTrain[0]))
	for i := range w {
		w[i] = rand.Float64() * 2
	}
	b := rand.Float64() * 2
	alpha := 1e-3
	epochs := 100000
	// Output formatting
	fmt.Printf("Start values of weights and bias: %v, %v: \n", w, b)
	w, b, dw, db = gradientDescent(xTrain, yTrain, w, alpha, b, epochs)
	fmt.Printf("End values of weights and bias: %v, %v: \n", w, b)
	fmt.Printf("End values of dw and db: %v, %v: \n", dw, db)
	fmt.Printf("Epochs: %v\n", epochs)
	score := accuracy(xTest, yTest, w, b)
	fmt.Printf("Score: %v\n", score)
	// drawing
	var drawR, drawG plotter.XYs
	for i := 0; i < len(yTrain); i++ {
		if yTrain[i] == 0 {
			drawR = append(drawR, struct{ X, Y float64 }{X: xTrain[i][0], Y: xTrain[i][1]})
		} else {
			drawG = append(drawG, struct{ X, Y float64 }{X: xTrain[i][0], Y: xTrain[i][1]})
		}
	}

	scatterR, err := plotter.NewScatter(drawR)
	if err != nil {
		panic(err)
	}
	scatterR.GlyphStyle.Color = color.RGBA{R: 255, A: 255}
	scatterR.GlyphStyle.Radius = vg.Points(4)
	p.Add(scatterR)

	scatterG, err := plotter.NewScatter(drawG)
	if err != nil {
		panic(err)
	}
	scatterG.GlyphStyle.Color = color.RGBA{G: 255, A: 255}
	scatterG.GlyphStyle.Radius = vg.Points(4)
	p.Add(scatterG)

	line, _ := plotter.NewLine(plotter.XYs{
		{X: 20, Y: (-w[0]*20 - b) / w[1]},
		{X: 100, Y: (-w[0]*100 - b) / w[1]}})
	line.Color = color.RGBA{B: 255, A: 255}
	p.Add(line)

	p.Title.Text = "LOGistic regression"
	p.X.Label.Text = "exam1"
	p.Y.Label.Text = "exam2"

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "scatter.png"); err != nil {
		panic(err)
	}
	// draw heatmap
	// special thanks to
	//https://medium.com/@balazs.dianiska/generating-heatmaps-with-go-83988b22c000
	// The only question is
	// what does the heatmap show?
	// how to tweak values for the best result?
	ph := plot.New()
	var plotData plottable
	plotData.grid = xTest
	plotData.N = len(xTest)
	plotData.M = len(xTest[0])
	plotData.resolution = 50
	plotData.minX = 0.2
	plotData.minY = 25
	pal := moreland.SmoothBlueRed().Palette(255)
	heatmap := plotter.NewHeatMap(plotData, pal)
	ph.Add(heatmap)
	if err := ph.Save(4*vg.Inch, 4*vg.Inch, "heatmap.png"); err != nil {
		panic(err)
	}
}
func (p plottable) Dims() (c, r int) {
	return p.N, p.M
}
func (p plottable) X(c int) float64 {
	return p.minX + float64(c)*p.resolution
}
func (p plottable) Y(r int) float64 {
	return p.minY + float64(r)*p.resolution
}
func (p plottable) Z(c, r int) float64 {
	return p.grid[c][r]
}
