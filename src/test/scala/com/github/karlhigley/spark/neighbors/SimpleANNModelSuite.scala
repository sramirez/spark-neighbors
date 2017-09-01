package com.github.karlhigley.spark.neighbors

import org.apache.spark.ml.linalg.Vector
import org.scalatest.{ BeforeAndAfterAll, BeforeAndAfter, FunSuite }
import com.github.karlhigley.spark.neighbors.ANNModel.IDPoint
import org.apache.spark.ml.feature.LabeledPoint

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


/**
 * @author Thomas Moerman
 */
@RunWith(classOf[JUnitRunner])
class SimpleANNModelSuite extends FunSuite with BeforeAndAfterAll {

  val numPoints = 1000
  val dimensions = 100
  val density = 0.5
  val nClasses = 3
  
  var points: Iterable[IDPoint] = _

  override def beforeAll() {
    val localPoints = TestHelpers.generateRandomLabeledPoint(numPoints, dimensions, density, nClasses)
    points = localPoints.zipWithIndex.map{ case (lp, i) => i.toLong -> lp}
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