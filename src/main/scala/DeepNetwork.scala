package net.deep

import java.awt.image.BufferedImage
import java.awt.{Color, Rectangle}
import java.io.{File, FileInputStream, FileOutputStream}
import java.io.{ObjectInputStream, ObjectOutputStream}
import javax.imageio.ImageIO

import scala.math.{exp, sqrt}
import scala.util.Random


object DeepNetwork extends App {
    val nn = new NeuralNetwork( List(2,1), learningRate = 0.1, Random.nextDouble )
    nn.train( Array(-1, 2), Array(2) )
    nn.train( Array(0, 2),  Array(2) )
    println(nn)
}

/**
* Constructor takes args: Int* where each number means number of neurons in a layer
*/
class NeuralNetwork(layersDef: List[Int], learningRate: Double, weightGen: () => Double) {
    var time: Int = 0
    val layers = new Array[Layer](layersDef.length + 1)

    def input = layers(1)
    def output = layers.last

    // CONSTRUCTOR
    // that's a fake layer so the input layer's neurons have one "input"
    this.layers(0) = new Layer(1, "linear", learningRate)
    // input layer needs to has linear activation to preserve the input values
    // all layers except the output layer has one extra neuron - the bias neuron
    this.layers(1) = new Layer(layersDef(0), this.layers(0), "linear", learningRate, withBias=true)
    // hiden layers and the outp layer are initialized here
    for (i <- 2 until layersDef.length) {
        this.layers(i) = new Layer(layersDef(i-1), this.layers(i-1), "sigmoid", learningRate, withBias=true, weightGen);
    }
    // output layer has no bias neuron 
    this.layers(layersDef.length) = new Layer(layersDef.last, this.layers(layersDef.length-1), "sigmoid", learningRate, withBias=false, weightGen);

    def train(x: Iterable[Double], t: Iterable[Double]) {
        this.forward(x)
        this.backward(t)
        this.time += 1
    }

    def forward(x: Iterable[Double]) {
        // set up inputs
        for(nx <- this.input.neurons zip x) nx._1.forward(0, nx._2)
        // and fire
        this.layers.tail.foreach(_.fire()) // tail for skipping the first fake layer
    }

    def backward(t: Iterable[Double]) {
        // we use square error E = 1/2 (t-y)**2  with derivative (y-t)
        // compute output errors (since we assign to gradient we use just dE/dy = y-t)
        for(ot <- this.output.neurons zip t) ot._1.gradient = ot._1.value - ot._2
        // and backpropagate (skip the output layer bcs we set it up manually)
        for(layer <- this.layers.reverse.tail) layer.backpropagate()
    }

    def classify(x: Iterable[Double]) : Array[Double] = {
        this.forward(x)
        this.output.neurons.map(_.value)  // collect the outputs
    }

    override def toString() = "digraph TIME%d {\n%s}".format(this.time, this.layers.tail.map(_.mkString).mkString("\n"))

    def toImage(file: File, squareSize: Int =10) {
        val dim = sqrt(input.neuronsCnt).toInt 
        val image = new BufferedImage(dim * squareSize * (output.neuronsCnt + 1), // include padding
                                      dim * squareSize,
                                      BufferedImage.TYPE_BYTE_GRAY)
        val graphics = image.createGraphics()
        for(outputNum <- 0 until output.neuronsCnt) {
            val abs_shift: Int = outputNum * (dim + 1) * squareSize
            for(w <- 0 until dim) {
                for(h <- 0 until dim) {
                    var shade = (input.neurons(w + (h*dim)).outputs(outputNum).weight * 255.0).toInt
                    shade = if(shade > 255) 255 else if(shade < 0) 0 else shade
                    graphics.setColor(new Color(shade, shade, shade))
                    graphics.fill(new Rectangle(abs_shift + w * squareSize, h * squareSize,
                        squareSize, squareSize))
                }
            }
        }
        ImageIO.write(image, "jpeg", file)
    }

    def toFile(file: File) {
        val oos = new ObjectOutputStream(new FileOutputStream(file))
        oos.writeObject(this)
        oos.close
    }
}

object NeuralNetwork {

    def fromFile(file: File): NeuralNetwork = {
        val ois = new ObjectInputStream(new FileInputStream(file))
        val nn = ois.readObject().asInstanceOf[NeuralNetwork]
        ois.close
        nn
    }

}


class Layer(val neuronsCnt: Int, val activation: String, learningRate: Double, val withBias: Boolean = true)  {
    var level: Int = 0
    val neurons = new Array[Neuron](neuronsCnt + {if(withBias) 1 else 0})
    if (withBias) {
        neurons(0) = Neuron(0, this.level, "linear", learningRate) // bias
        neurons(0).inputs = Array[Double](1.0) // bias has one, constant input
        for (i <- 1 to neuronsCnt ) { neurons(i) = Neuron(i, this.level, this.activation, learningRate) }
    }
    for (i <- 0 until neuronsCnt ) { neurons(i) = Neuron(i, this.level, this.activation, learningRate) }

    def this(neuronsCnt: Int, previous: Layer, activation: String, learningRate: Double, withBias: Boolean = true, weightGen: () => Double = Random.nextDouble) {
        this(neuronsCnt, activation, learningRate, withBias)
        // update level of current layer
        this.level = previous.level + 1
        this.neurons.foreach( _.level = this.level )
        // initialize inputs in current neurons (bcs we know the count now)
        this.neurons.foreach( _.link_from(previous.neurons) )
        // create links from the previous layer to the current
        if(withBias) previous.neurons.tail.foreach( _.link_to(this.neurons, weightGen) )
        else previous.neurons.foreach( _.link_to(this.neurons, weightGen) )
    }

    def apply(index: Int) = neurons(index)

    def fire() {
        this.neurons.foreach(_.fire)
    }

    def backpropagate() {
        this.neurons.foreach(_.backward)
    }

    override def toString() = "l%d".format(level)

    def mkString() = this.neurons.map(_.mkString).mkString("\n")

}


object Neuron {
    def apply(position: Int, level: Int, activation: String, learningRate: Double): Neuron = 
        activation match {
            case "linear" => new Neuron(position, level, learningRate) with LinearActivation
            case "sigmoid" => new Neuron(position, level, learningRate) with SigmoidActivation
            case _ => new Neuron(position, level, learningRate) with SigmoidActivation
        }
}


abstract class Neuron(val position: Int, var level: Int = 0, val learningRate: Double) {
    class Link(val neuron: Neuron, var weight: Double=Random.nextDouble() + 0.001)

    var inputs = Array.empty[Double]
    var outputs = Array.empty[Link]

    var value: Double = 1.0
    var gradient: Double = 0.0

    def forward (position: Int, value: Double) {
        inputs(position) = value
    }

    def fire () {
        for( link <- this.outputs ) link.neuron.forward(this.position, this.value * link.weight)
    }

    def backward () {
        var dw_i: Double = 0.0
        this.gradient = 0.0
        for( link <- this.outputs ) {
            // update own gradient: dE/dyi = SUM_{j \in J}{(dE/dy_j)}
            this.gradient += link.weight * link.neuron.gradient
            // now update output's weight using it's gradient and own value (it's input value)
            // dE/dw =   dz/dw    *         dy/dz          *      dE/dy
            dw_i     = this.value * link.neuron.derivative * link.neuron.gradient
            // update output's weight using learningRate
            link.weight -= dw_i * link.neuron.learningRate
        }
    }

    def derivative() : Double

    def link_from(neurons: Array[Neuron]) {
        this.inputs = new Array[Double](neurons.length)
    }

    def link_to(neurons: Array[Neuron], weightGen: () => Double) {
        this.outputs = new Array[Link](neurons.length)
        for(i <- 0 until neurons.length) this.outputs(i) = new Link(neurons(i), weightGen())
    }

    override def toString() = "\"n%d_%d=%.2f/%.2f\"".format(this.level, this.position, this.value, this.gradient)

    def mkString() = this.outputs.map(o => "%s -> %s [label=\"%.2f\" weight=%d]".format(this.toString, o.neuron, o.weight, (o.weight*3).toInt )).mkString("\n")

}


trait SigmoidActivation extends Neuron {
    override def fire() {
        this.value = 1.0 / (1.0 + exp(-1.0 * inputs.sum))
        super.fire()
    }

    def derivative() : Double = this.value * (1.0 - this.value)
}


trait LinearActivation extends Neuron {
    override def fire() {
        this.value = inputs.sum
        super.fire()
    }

    def derivative() : Double = 1.0
}
