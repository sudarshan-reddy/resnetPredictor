package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func main() {
	imgName := os.Args[1]
	model, err := tf.LoadSavedModel("forGo", []string{"tags"}, nil)
	if err != nil {
		log.Fatal(err)
	}

	imageFile, err := os.Open(imgName)
	if err != nil {
		log.Fatal(err)
	}
	var imgBuffer bytes.Buffer
	io.Copy(&imgBuffer, imageFile)
	img, err := readImage(&imgBuffer, "jpg")
	if err != nil {
		log.Fatal("error making tensor: ", err)
	}

	result, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("input_1").Output(0): img,
		},
		[]tf.Output{
			model.Graph.Operation("inferenceLayer/Softmax").Output(0),
		},
		nil,
	)

	if err != nil {
		log.Fatal(err)
	}

	if preds, ok := result[0].Value().([][]float32); ok {
		fmt.Println(preds)
		if preds[0][0] > preds[0][1] {
			fmt.Println("male")
		} else {
			fmt.Println("female")
		}
	}
}

func readImage(imageBuffer *bytes.Buffer, imageFormat string) (*tf.Tensor, error) {
	tensor, err := tf.NewTensor(imageBuffer.String())
	if err != nil {
		return nil, err
	}
	graph, input, output, err := transformGraph(imageFormat)
	if err != nil {
		return nil, err
	}
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

func transformGraph(imageFormat string) (graph *tf.Graph, input,
	output tf.Output, err error) {
	const (
		H, W  = 224, 224
		Mean  = float32(117)
		Scale = float32(1)
	)
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)

	var decode tf.Output
	switch imageFormat {
	case "png":
		decode = op.DecodePng(s, input, op.DecodePngChannels(3))
	case "jpg",
		"jpeg":
		decode = op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))
	default:
		return nil, tf.Output{}, tf.Output{},
			fmt.Errorf("imageFormat not supported: %s", imageFormat)
	}

	output = op.Div(s,
		op.Sub(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.Cast(s, decode, tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()
	return graph, input, output, err
}
