package com.github.karlhigley.spark.neighbors

import org.apache.spark.mllib.linalg.{ Vector => MLLibVector }
import org.scalatest.{ BeforeAndAfterAll, BeforeAndAfter, FunSuite }

/**
 * @author Thomas Moerman
 */
class SimpleANNModelSuite extends FunSuite with BeforeAndAfterAll {

  val numPoints = 1000
  val dimensions = 100
  val density = 0.5

  var points: Iterable[(Long, MLLibVector)] = _

  override def beforeAll() {
    val localPoints = TestHelpers.generateRandomPoints(numPoints, dimensions, density)
    points = localPoints.zipWithIndex.map { case (v, idx) => (idx.toLong, v) }
  }

  test("average selectivity is between zero and one") {
    val ann =
      new ANN(dimensions, "cosine")
        .setTables(1)
        .setSignatureLength(16)

    val model = ann.train(points)
    val selectivity = model.avgSelectivity()

    assert(selectivity > 0.0)
    assert(selectivity < 1.0)
  }

  test("average selectivity increases with more tables") {
    val ann =
      new ANN(dimensions, "cosine")
        .setTables(1)
        .setSignatureLength(16)

    val model1 = ann.train(points)

    ann.setTables(2)
    val model2 = ann.train(points)

    assert(model1.avgSelectivity() < model2.avgSelectivity())
  }

  test("average selectivity decreases with signature length") {
    val ann =
      new ANN(dimensions, "cosine")
        .setTables(1)
        .setSignatureLength(4)

    val model4 = ann.train(points)

    ann.setSignatureLength(8)
    val model8 = ann.train(points)

    assert(model4.avgSelectivity() > model8.avgSelectivity())
  }

}