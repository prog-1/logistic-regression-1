package main

import (
	"image"
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

const (
	lineXMin, lineXMax = 25, 100 // x coordinates of the line
)

// Converting plot to ebiten.Image
func PlotToImage(p *plot.Plot) *ebiten.Image {

	img := image.NewRGBA(image.Rect(0, 0, sW, sH)) //creating image.RGBA to store the plot

	c := vgimg.NewWith(vgimg.UseImage(img)) //creating plot drawer for the image

	p.Draw(draw.New(c)) //drawing plot on the image

	return ebiten.NewImageFromImage(c.Image()) //converting image.RGBA to ebiten.Image (doing in Draw)
	///Black screen issue: was giving "img" instead of "c.Image()" in the function.
}

//###################################################################################

func (a *App) updatePlot(w []float64, b float64, inputs [][]float64, y []float64) {

	//################# Initialization ##########################

	p := plot.New() //initializing plot

	//#################### Points ##############################

	//Colors
	blue := color.RGBA{0, 0, 255, 255}
	red := color.RGBA{255, 0, 0, 255}

	//Plotters
	var greenPlotter plotter.XYs //initializing green point plotter
	var redPlotter plotter.XYs   //initializing red point plotter

	//Distributing the points to separate plotters
	for i := 0; i < numberOfPoints; i++ { //for every point
		if y[i] == 0 { //if the current point is false/0/negative
			greenPlotter = append(greenPlotter, plotter.XY{X: inputs[i][0], Y: inputs[i][1]}) //Saving the point in green plotter
		} else { //if the current point is true/1/positive
			redPlotter = append(redPlotter, plotter.XY{X: inputs[i][0], Y: inputs[i][1]}) //Saving the point in red plotter
		}
	}

	//Green scatter
	blueScatter, _ := plotter.NewScatter(greenPlotter) //creating new scatter from point data
	blueScatter.Color = blue
	p.Add(blueScatter)

	//Red scatter
	redScatter, _ := plotter.NewScatter(redPlotter) //creating new scatter from point data
	redScatter.Color = red
	p.Add(redScatter)

	//####################### Line ##############################

	/*
		(x1 = X | x2 = Y)

		0 = k*x + b
		0 = w1x1 + w2x2 + b
		-w2x2 = w1x1 + b
		w2x2 = - b - w1x1
		x2 = (-w1x1-b) / w2
	*/

	linePlotter := plotter.XYs{
		{X: lineXMin, Y: (-w[0]*lineXMin - b) / w[1]},
		{X: lineXMax, Y: (-w[0]*lineXMax - b) / w[1]},
	}

	line, _ := plotter.NewLine(linePlotter) //creating line
	p.Add(line)                             // adding line to the plot

	//##################### Ebiten #############################

	a.plot = p //replacing old plot with new one

}
