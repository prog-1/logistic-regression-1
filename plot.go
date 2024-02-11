package main

import (
	"image"
	"image/color"

	"github.com/hajimehoshi/ebiten"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

const (
	lineMin, lineMax = 0, 120 //line lenght
)

// Converting plot to ebiten.Image
func PlotToImage(p *plot.Plot) *ebiten.Image {

	img := image.NewRGBA(image.Rect(0, 0, sW, sH)) //creating image.RGBA to store the plot

	c := vgimg.NewWith(vgimg.UseImage(img)) //creating plot drawer for the image

	p.Draw(draw.New(c)) //drawing plot on the image

	return ebiten.NewImageFromImage(c.Image()) //converting image.RGBA to ebiten.Image (doing in Draw)
	///Black screen issue: was giving "img" instead of "c.Image()" in the function.
}

// Returns new plot with given data
func (a *App) updatePlot(w []float64, b float64, px, py [][]float64) {

	//################# Initialization ##########################

	p := plot.New() //initializing plot

	//Group colors
	color1 := color.RGBA{150, 0, 0, 255}
	color2 := color.RGBA{0, 150, 0, 255}

	//##################### Line ##############################

	//Line points
	lp := plotter.XYs{
		{X: lineMin, Y: calculateY(lineMin, w, b)},
		{X: lineMax, Y: calculateY(lineMax, w, b)},
	}

	line, _ := plotter.NewLine(lp) //creating line

	p.Add(line) //adding line to the plot

	//#################### Points ##############################

	// Group 1

	var points1 plotter.XYs //initializing point plotter

	for i := 0; i < len(px); i++ { //for every point in group
		points1 = append(points1, plotter.XY{X: px[i][0], Y: px[i][1]}) //Saving all points in plotter
	}
	scatter1, _ := plotter.NewScatter(points1) //creating new scatter from point data°
	scatter1.Color = color1

	p.Add(scatter1) //adding points to plot

	// Group 2

	var points2 plotter.XYs //initializing point plotter

	for i := 0; i < len(px); i++ { //for every point in group
		points2 = append(points2, plotter.XY{X: py[i][0], Y: py[i][1]}) //Saving all points in plotter
	}
	scatter2, _ := plotter.NewScatter(points2) //creating new scatter from point data°
	scatter2.Color = color2

	p.Add(scatter2) //adding points to plot

	//##################### App #############################

	a.plot = p //replacing old plot with new one
}

func calculateY(x, w []float64, b float64) float64 {
	return dot(x, w) + b
}
