package example

import breeze.linalg._

object Main extends App {

  type Y = DenseMatrix[Double]
  type X = DenseMatrix[Double]
  type Data = List[(X, Y)]

  val mnist = Mnist.trainDataset
  val allData = mnist.examples.map {
    case (x, y) => (x.toDenseMatrix.t, y.toDenseMatrix.t)
  }.toList
  val (trainData, testData) = allData.splitAt(50000)
  val (firstX, firstY) = trainData.head
  val nn = NN.init(List(firstX.rows, 100, firstY.rows))
  Util.time(
    NN.sgd(nn,
           trainingData = trainData,
           epochs = 30,
           batchSize = 10,
           learningRate = 0.5,
           testData = testData))
}
