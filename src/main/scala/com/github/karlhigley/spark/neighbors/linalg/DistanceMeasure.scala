package com.github.karlhigley.spark.neighbors.linalg

import breeze.linalg.norm
import org.apache.spark.ml.linalg.{SparseVector, Vectors, Vector}

import org.apache.spark.ml.linalg.LinalgShim

/**
 * This abstract base class provides the interface for
 * distance measures to be used in computing the actual
 * distances between candidate pairs.
 *
 * It's framed in terms of distance rather than similarity
 * to provide a common interface that works for Euclidean
 * distance along with other distances. (Cosine distance is
 * admittedly not a proper distance measure, but is computed
 * similarly nonetheless.)
 */
sealed abstract class DistanceMeasure extends Serializable {
  def apply(v1: Vector, v2: Vector): Double
}

final object CosineDistance extends DistanceMeasure {

  /**
   * Compute cosine distance between vectors
   *
   * LinalgShim reaches into Spark's private linear algebra
   * code to use a BLAS dot product. Could probably be
   * replaced with a direct invocation of the appropriate
   * BLAS method.
   */
  def apply(v1: Vector, v2: Vector): Double = {
    val dotProduct = LinalgShim.dot(v1, v2)
    val norms = Vectors.norm(v1, 2) * Vectors.norm(v2, 2)
    1.0 - (math.abs(dotProduct) / norms)
  }
}

final object EuclideanDistance extends DistanceMeasure {

  def apply(v1: Vector, v2: Vector): Double = {
    val b1 = LinalgShim.toBreeze(v1)
    val b2 = LinalgShim.toBreeze(v2)
    norm(b1 - b2, 2.0)
  }

}

final object ManhattanDistance extends DistanceMeasure {

  def apply(v1: Vector, v2: Vector): Double = {
    val b1 = LinalgShim.toBreeze(v1)
    val b2 = LinalgShim.toBreeze(v2)
    norm(b1 - b2, 1.0)
  }

}

final object FractionalDistance extends DistanceMeasure {

  def apply(v1: Vector, v2: Vector): Double = {
    val b1 = LinalgShim.toBreeze(v1)
    val b2 = LinalgShim.toBreeze(v2)
    norm(b1 - b2, 0.5)
  }

}

final object HammingDistance extends DistanceMeasure {

  /**
   * Compute Hamming distance between vectors
   *
   * Since MLlib doesn't support binary vectors, this uses
   * sparse vectors and considers any active (i.e. non-zero)
   * index to represent a set bit
   */
  def apply(v1: Vector, v2: Vector): Double = {
    val i1 = v1.asInstanceOf[SparseVector].indices.toSet
    val i2 = v2.asInstanceOf[SparseVector].indices.toSet
    (i1.union(i2).size - i1.intersect(i2).size).toDouble
  }

}

final object JaccardDistance extends DistanceMeasure {

  /**
   * Compute Jaccard distance between vectors
   *
   * Since MLlib doesn't support binary vectors, this uses
   * sparse vectors and considers any active (i.e. non-zero)
   * index to represent a member of the set
   */
  def apply(v1: Vector, v2: Vector): Double = {
    val indices1 = v1.asInstanceOf[SparseVector].indices.toSet
    val indices2 = v2.asInstanceOf[SparseVector].indices.toSet
    val intersection = indices1.intersect(indices2)
    val union = indices1.union(indices2)
    1.0 - (intersection.size / union.size.toDouble)
  }

}