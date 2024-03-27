package main

import (
	"image"
	"image/color"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

func Plot(legend string, ps ...plot.Plotter) *image.RGBA {
	p := plot.New()
	p.Add(append([]plot.Plotter{
		plotter.NewGrid(),
	}, ps...)...)
	p.Legend.Add(legend)
	img := image.NewRGBA(image.Rect(0, 0, screenWidth, screenHeight))
	c := vgimg.NewWith(vgimg.UseImage(img))
	p.Draw(draw.New(c))
	return c.Image().(*image.RGBA)
}

func FunctionWithColor(f func(x float64) float64, color color.RGBA) *plotter.Function {
	res := plotter.NewFunction(f)
	res.Color = color
	return res
}
