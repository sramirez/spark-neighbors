package com.github.karlhigley.spark.neighbors

import scala.util.Random
import org.apache.spark.ml.linalg.{ Vector => MLLibVector, SparseVector }
import org.apache.spark.ml.feature.LabeledPoint

object TestHelpers {
  
  def generateRandomLabeledPoint(quantity: Int, dimensions: Int, density: Double, nClasses: Int) = {
    val vectors = generateRandomPoints(quantity, dimensions, density)
    vectors.map{v => new LabeledPoint(Random.nextInt(nClasses), v)}
  }

  def generateRandomPoints(quantity: Int, dimensions: Int, density: Double) = {
    val numElements = math.floor(dimensions * density).toInt
    val points = new Array[MLLibVector](quantity)
    var i = 0
    while (i < quantity) {
      val indices = generateIndices(numElements, dimensions)
      val values = generateValues(numElements)
      points(i) = new SparseVector(dimensions, indices, values)
      i += 1
    }
    points
  }

  def generateIndices(quantity: Int, dimensions: Int) = {
    val indices = new Array[Int](quantity)
    var i = 0
    while (i < quantity) {
      val possible = Random.nextInt(dimensions)
      if (!indices.contains(possible)) {
        indices(i) = possible
        i += 1
      }
    }
    indices.sorted
  }

  def generateValues(quantity: Int) = {
    val values = new Array[Double](quantity)
    var i = 0
    while (i < quantity) {
      values(i) = Random.nextGaussian()
      i += 1
    }
    values
  }

}
