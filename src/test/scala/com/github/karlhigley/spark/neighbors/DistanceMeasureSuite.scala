package com.github.karlhigley.spark.neighbors

import org.scalatest.FunSuite

import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}

import com.github.karlhigley.spark.neighbors.linalg._

class DistanceMeasureSuite extends FunSuite with TestSparkContext {
  import org.scalactic.Tolerance._

  val values = Array(1.0, 1.0, 1.0, 1.0)

  val v1 = new SparseVector(10, Array(0, 3, 6, 8), values)
  val v2 = new SparseVector(10, Array(1, 4, 7, 9), values)
  val v3 = new SparseVector(10, Array(2, 5, 7, 9), values)

  val v4 = new DenseVector(Array(1, 0, 0, 1, 0, 0, 1, 0, 1, 0).map(_.toDouble))
  val v5 = new DenseVector(Array(0, 1, 0, 0, 1, 0, 0, 1, 0, 1).map(_.toDouble))
  val v6 = new DenseVector(Array(0, 0, 1, 0, 0, 1, 0, 1, 0, 1).map(_.toDouble))

  test("Cosine distance") {
    assert(CosineDistance(v1, v1) === 0.0)
    assert(CosineDistance(v1, v2) === 1.0)
    assert(CosineDistance(v2, v3) === 0.5)

    assert(CosineDistance(v4, v4) === 0.0)
    assert(CosineDistance(v4, v5) === 1.0)
    assert(CosineDistance(v5, v6) === 0.5)
  }

  test("Euclidean distance") {
    assert(EuclideanDistance(v1, v1) === 0.0)
    assert(EuclideanDistance(v1, v2) === 2.83 +- 0.01)
    assert(EuclideanDistance(v2, v3) === 2.0)

    assert(EuclideanDistance(v4, v4) === 0.0)
    assert(EuclideanDistance(v4, v5) === 2.83 +- 0.01)
    assert(EuclideanDistance(v5, v6) === 2.0)
  }

  test("Manhattan distance") {
    assert(ManhattanDistance(v1, v1) === 0.0)
    assert(ManhattanDistance(v1, v2) === 8.0)
    assert(ManhattanDistance(v2, v3) === 4.0)

    assert(ManhattanDistance(v4, v4) === 0.0)
    assert(ManhattanDistance(v4, v5) === 8.0)
    assert(ManhattanDistance(v5, v6) === 4.0)
  }

  test("Fractional distance") {
    assert(FractionalDistance(v1, v1) === 0.0)
    assert(FractionalDistance(v1, v2) === 64.0)
    assert(FractionalDistance(v2, v3) === 16.0)

    assert(FractionalDistance(v4, v4) === 0.0)
    assert(FractionalDistance(v4, v5) === 64.0)
    assert(FractionalDistance(v5, v6) === 16.0)
  }

  test("Hamming distance") {
    assert(HammingDistance(v1, v1) === 0.0)
    assert(HammingDistance(v1, v2) === 8.0)
    assert(HammingDistance(v2, v3) === 4.0)
  }

  test("Jaccard distance") {
    assert(JaccardDistance(v1, v1) === 0.0)
    assert(JaccardDistance(v1, v2) === 1.0)
    assert(JaccardDistance(v2, v3) === 0.67 +- 0.01)
  }

}