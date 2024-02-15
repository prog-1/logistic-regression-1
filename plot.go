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

	red := color.RGBA{150, 0, 0, 255}
	//green := color.RGBA{0, 0, 150, 255}

	//#################### Points ##############################

	var points plotter.XYs //initializing point plotter

	for i := 0; i < numberOfPoints; i++ { //for every point
		points = append(points, plotter.XY{X: inputs[i][0], Y: inputs[i][1]}) //Saving all points in plotter
	}
	scatter, _ := plotter.NewScatter(points) //creating new scatter from point data
	scatter.Color = red

	p.Add(scatter)

	//##################### Ebiten #############################

	a.plot = p //replacing old plot with new one

}
