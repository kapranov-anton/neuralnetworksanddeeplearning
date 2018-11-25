package example

import java.time.Duration

object Util {
  def time[R](block: => R): R = {
    val t0 = System.currentTimeMillis()
    val result = block // call-by-name
    val t1 = System.currentTimeMillis()
    println("Elapsed time: " + Duration.ofMillis(t1 - t0))
    result
  }

  def zipWith[A, B, C](a: List[A], b: List[B], op: (A, B) => C): List[C] =
    a.zip(b).map(p => op(p._1, p._2))

  def slidingWindow[A](list: List[A]): List[(A, A)] = list.init.zip(list.tail)
}

