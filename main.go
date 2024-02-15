package main

import (
	"encoding/csv"
	"image/color"
	"image"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
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

func split(data [][]string) (xTrain, xTest [][]float64, yTrain, yTest []float64, draw plotter.XYs) {
	half := len(data) / 2
	draw = make(plotter.XYs, half)
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
			draw[i].X, _ = strconv.ParseFloat(row[0], 64)
			draw[i].Y, _ = strconv.ParseFloat(row[1], 64)
		}
		yTest[i], _ = strconv.ParseFloat(row[2], 64)
	}
	//fmt.Println(xTrain, xTest, yTrain, yTest)
	return xTrain, xTest, yTrain, yTest, draw
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
	xTrain, xTest, yTrain, yTest, draw := split(data)
	w := make([]float64, len(xTrain[0]))
	for i := range w {
		w[i] = rand.Float64()*2
	}
	b := rand.Float64() *2
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
	scatter, err := plotter.NewScatter(draw)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Color = color.RGBA{R: 255, A: 255}
	scatter.GlyphStyle.Radius = vg.Points(4)
	// Add the scatter plot to the plot and set the axes labels
	p.Add(scatter)
	p.Title.Text = "LOGistic regression"
	p.X.Label.Text = "exam1"
	p.Y.Label.Text = "exam2"

// Save the plot to a PNG file
if err := p.Save(4*vg.Inch, 4*vg.Inch, "scatter.png"); err != nil {
    panic(err)
}
}

func Plot(ps ...plot.Plotter) *image.RGBA {
	p := plot.New()
	p.Add(append([]plot.Plotter{
		plotter.NewGrid(),
	}, ps...)...)
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	c := vgimg.NewWith(vgimg.UseImage(img))
	p.Draw(draw.New(c))
	return c.Image().(*image.RGBA)
}