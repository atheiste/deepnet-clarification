package classification

import net.deep.NeuralNetwork
import java.io.{File, FileInputStream, DataInputStream}
import java.awt.image.BufferedImage
import javax.imageio.ImageIO


object MNIST extends App{

    if (args.length < 4) {
        println("usage: <MNIST> train_images_file train_labels_file test_images_file test_labels_file")
        System.exit(1)
    }
    var images = new DataInputStream(new FileInputStream(args(0)))
    var labels = new DataInputStream(new FileInputStream(args(1)))

    images.readInt // read out the BOM
    labels.readInt // read out the BOM

    var labelsCnt = labels.readInt
    var imagesCnt = images.readInt
    var width = images.readInt
    var height = images.readInt

    if (labelsCnt != imagesCnt) {
        Console.withOut(Console.err){println("Number of images and labels does not match")}
        System.exit(1)
    }

    println("Will train on %d images of size %dx%d".format(imagesCnt, width, height))

    val nn = {
        val file = new File("export/nn.dat")
        if (file.exists)
            NeuralNetwork.fromFile(file)
        else
            new NeuralNetwork(List(width*height, 10), learningRate=0.01, () => Double = 0.01)
    }
    var buffer = new Array[Byte](width*height)
    var label: Int = 255
    var limit = 3500

    for(i <- 1 to limit) {
        images.read(buffer)
        label = labels.readByte.toInt
        for(j <- 1 to 200) {
            // linear transformation of inputs/output into double <0,1>
            nn.train(
                buffer.map(_.toDouble / 255.0),            // inputs (normalized)
                0 to 9 map(x => {if (x == label) 1.0 else 0.0}.toDouble)   // output (byte to "1 out of N" code)
            )
            // if(j % 100 == 0) {
            //     nn.toImage(new File("net%d.jpeg".format(j)))
            //     println("Errors: " + nn.output.neurons.map(_.gradient).map("%.3f".format(_)).mkString(", "))
            // }
        }
        // println("Expect: " + 0.to(9).map(x => {if (x == label) 1.0 else 0.0}.toDouble).mkString(", "))
        // println("Values: " + nn.output.neurons.map(_.value).map("%.3f".format(_)).mkString(", "))
        if (i % 500 == 0) {
            println(String.valueOf(i))
            nn.toImage(new File("export/weights%d.jpeg".format(i)))
        }
    }
    nn.toFile(new File("export/nn.dat"))

    images.close
    labels.close

    // CLASSIFY phase
    images = new DataInputStream(new FileInputStream(args(2)))
    labels = new DataInputStream(new FileInputStream(args(3)))

    images.readInt // read out the BOM
    labels.readInt // read out the BOM

    labelsCnt = labels.readInt
    imagesCnt = images.readInt
    width = images.readInt
    height = images.readInt

    if (labelsCnt != imagesCnt) {
        Console.withOut(Console.err){println("Number of images and labels does not match")}
        System.exit(1)
    }

    println("Will classify on %d images of size %dx%d".format(imagesCnt, width, height))

    var y : Array[Double] = null
    var correct: Int = 0
    var lastMax: Double = -1.0
    var maxIndex: Int = 0

    limit = 200

    for(i <- 1 to limit) {
        lastMax = -1.0
        maxIndex = 0

        images.read(buffer)
        label = labels.readByte.toInt

        // linear transformation of inputs/output into double <0,1>
        y = nn.classify(buffer.map(_.toDouble / 255.0))            // inputs (normalized) )
        if (i % 100 == 0) println(String.valueOf(i))

        for(j <- 0 until y.length) {
            if (y(j) >= lastMax) {
                maxIndex = j
                // println("^^ %.3f (y) > %.3f (max) therefor maxIndex is %d (as j=%d)".format(y(j), lastMax, maxIndex, j))
                lastMax = y(j)
            } else {
                // println("## %.3f (y) < %.3f (max) therefor maxIndex stays %d (as j=%d)".format(y(j), lastMax, maxIndex, j))
            }
        }
        println(y.map("%.4f".format(_)).mkString(" ; "))
        println("Maximal index %d (correct %d) and that's %s".format(maxIndex, label, if (maxIndex == label) "RIGHT" else "WRONG"))
        if (maxIndex == label) correct += 1
    }

    println("Success %.2f percent by %d out of %d".format(correct / (limit.toDouble / 100.0), correct, limit))
    // val image = new  BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    // image.getRaster().setDataElements(0, 0, width, height, buffer)
    // ImageIO.write(image, "jpg", new File("sample.jpeg"))

    images.close
}