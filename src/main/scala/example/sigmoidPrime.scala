package example

import breeze.generic.{MappingUFunc, UFunc}
import breeze.numerics.sigmoid

/**
  * Derivative of the sigmoid function: s(x) * (1 - s(x))
  */
object sigmoidPrime extends UFunc with MappingUFunc {
  implicit object sigmoidImplDouble extends Impl[Double, Double] {
    def apply(x:Double): Double = sigmoid(x) * (1d - sigmoid(x))
  }
}
