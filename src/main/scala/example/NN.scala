package example

import breeze.linalg._
import breeze.linalg.operators.OpAdd
import breeze.numerics._
import breeze.stats.distributions.RandBasis

import scala.util.Random

case class NN(layers: List[Int],
              biases: List[NN.Matrix],
              weights: List[NN.Matrix]) {
  def bZipW: List[(NN.Matrix, NN.Matrix)] = biases.zip(weights)
}

object NN {
  type Matrix = DenseMatrix[Double]
  type Y = Matrix
  type X = Matrix
  type Data = List[(X, Y)]

  def init(layers: List[Int]): NN = {
    val seeded = RandBasis.withSeed(1)
    val biases = layers.tail.map(DenseMatrix.rand(_, 1, seeded.gaussian))
    val weights = Util.slidingWindow(layers).map {
      case (prevLayer, nextLayer) =>
        DenseMatrix.rand(nextLayer, prevLayer, seeded.gaussian)
    }
    NN(layers, biases, weights)
  }

  def feedforward(nn: NN, input: Matrix): Matrix = {
    nn.bZipW
      .foldLeft(input) {
        case (a, (b, w)) => sigmoid((w * a) + b)
      }
  }

  def evaluate(nn: NN, testData: NN.Data): Int =
    testData.count {
      case (x, y) =>
        argmax(feedforward(nn, x)) == argmax(y)
    }

  def recalcWeights(weights: List[Matrix],
                    nabla: List[Matrix],
                    learningRate: Double): List[Matrix] =
    weights.zip(nabla).map { case (w, nw) => w - (learningRate *:* nw) }

  def sumDeltas(a: List[Matrix], b: List[Matrix]): List[Matrix] =
    Util.zipWith(a, b, OpAdd.apply[Matrix, Matrix, Matrix])

  def updateBatch(nn: NN,
                  batch: List[(NN.X, NN.Y)],
                  learningRate: Double): NN = {
    val (nbs, nws) = batch.par.map { case (x, y) => backprop(nn, x, y) }.unzip
    val nablaB = nbs.reduce(sumDeltas)
    val nablaW = nws.reduce(sumDeltas)
    val weights = recalcWeights(nn.weights, nablaW, learningRate)
    val biases = recalcWeights(nn.biases, nablaB, learningRate)

    nn.copy(weights = weights, biases = biases)
  }

  def backprop(nn: NN, x: NN.X, y: NN.Y): (List[Matrix], List[Matrix]) = {
    val (activations, _ :: zs) =
      nn.bZipW
        .scanLeft(x -> x /* У входного слоя нет z, это значение заглушка*/ ) {
          case ((prevActivation, _), (b, w)) =>
            val z = (w * prevActivation) + b
            val activation = sigmoid(z)
            activation -> z
        }
        .unzip

    val costDerivative = activations.last - y
    var delta = costDerivative *:* sigmoidPrime(zs.last)

    var nablaB: Array[Matrix] = nn.biases.map(_ * 0.0).toArray
    var nablaW: Array[Matrix] = nn.weights.map(_ * 0.0).toArray
    nablaB(nablaB.length - 1) = delta
    nablaW(nablaW.length - 1) = delta * activations(activations.size - 2).t
    2.until(nn.layers.size).foreach { l =>
      val z = zs(zs.size - l)
      val sp = sigmoidPrime(z)
      delta = (nn.weights(nn.weights.size - l + 1).t * delta) *:* sp
      nablaB(nablaB.length - l) = delta
      nablaW(nablaW.length - l) = delta * activations(activations.size - l - 1).t
    }

    nablaB.toList -> nablaW.toList
  }

  def sgd(initNN: NN,
          trainingData: Data,
          epochs: Int,
          batchSize: Int,
          learningRate: Double,
          testData: Data): Unit = {
    val rand = new Random(17)

    val batches = for {
      epochCount <- 1.to(epochs).toIterator
      batch <- rand.shuffle(trainingData).grouped(batchSize)
    } yield epochCount -> batch

    val (_, model) = batches.foldLeft(0 -> initNN) {
      case ((prevEpoch, nn), (epochCount, batch)) =>
        if (prevEpoch != epochCount) {
          if (testData.nonEmpty) {
            println(f"Epoch $epochCount: ${accuracy(nn, testData)}%2.2f%%")
          } else {
            println(s"Epoch $epochCount complete")
          }
        }

        epochCount -> NN.updateBatch(nn, batch, learningRate / batch.size)
    }
    println("-" * 80)
    println(f"Final: ${accuracy(model, testData)}%2.2f%%")
  }

  def accuracy(nn: NN, testData: Data): Double =
    evaluate(nn, testData).toDouble * 100 / testData.size
}
