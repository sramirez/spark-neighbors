package com.github.karlhigley.spark.neighbors

import java.util.{Random => JavaRandom}
import com.github.karlhigley.spark.neighbors.collision.{BandingCollisionStrategy, SimpleCollisionStrategy}
import com.github.karlhigley.spark.neighbors.linalg._
import com.github.karlhigley.spark.neighbors.lsh.ScalarRandomProjectionFunction.{generateFractional, generateL1, generateL2}
import com.github.karlhigley.spark.neighbors.lsh._
import org.apache.spark.ml.linalg.{Vector => MLLibVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK
import scala.util.Random
import org.apache.spark.ml.feature.LabeledPoint
import com.github.karlhigley.spark.neighbors.ANNModel.IDPoint

/**
 * Approximate Nearest Neighbors (ANN) using locality-sensitive hashing (LSH)
 *
 * @see [[https://en.wikipedia.org/wiki/Nearest_neighbor_search Nearest neighbor search (Wikipedia)]]
 */
class fastANN private (
    private var origDimension: Int,
    private var numTables: Int,
    private var signatureLength: Int,
    private var randomSeed: Int
) {

  /**
   * Constructs an ANN instance with default parameters.
   */
  def this(dimensions: Int) = {
    this(
      origDimension = dimensions,
      numTables = 1,
      signatureLength = 128,
      randomSeed = Random.nextInt()
    )
  }

  /**
   * Number of hash tables to compute
   */
  def getTables(): Int = {
    numTables
  }

  /**
   * Number of hash tables to compute
   */
  def setTables(tables: Int): this.type = {
    numTables = tables
    this
  }

  /**
   * Number of elements in each signature (e.g. # signature bits for sign-random-projection)
   */
  def getSignatureLength(): Int = {
    signatureLength
  }

  /**
   * Number of elements in each signature (e.g. # signature bits for sign-random-projection)
   */
  def setSignatureLength(length: Int): this.type = {
    signatureLength = length
    this
  }

  /**
   * Random seed used to generate hash functions
   */
  def getRandomSeed(): Int = {
    randomSeed
  }

  /**
   * Random seed used to generate hash functions
   */
  def setRandomSeed(seed: Int): this.type = {
    randomSeed = seed
    this
  }
  
  /**
   * Build a fast ANN model using the given dataset.
   * Improvements in searches are based on (2013, Marukatat).
   * Instances are pre-indexed by norm, euclidean distance is
   * approximated by using the hamming distance of bit signatures.
   *
   * @param points    RDD of vectors paired with IDs.
   *                   IDs must be unique and >= 0.
   * @return fastANNModel containing computed hash tables
   */
  def fastANNtrain(points: RDD[IDPoint],
            nClasses: Int,
            persistenceLevel: StorageLevel = MEMORY_AND_DISK,
            thDistance: Float = .9f): fastANNModel = {

    val random = new JavaRandom(randomSeed)

    /** Only one table to reduce the complexity and hardness of computations */
    val hashFunctions: Seq[LSHFunction[_]] = Seq(
      SignRandomProjectionFunction.generate(origDimension, signatureLength, random))

    fastANNModel.train(
      points,
      hashFunctions,
      collisionStrategy = SimpleCollisionStrategy,
      measure = CosineDistance,
      signatureLength,
      nClasses,
      persistenceLevel
    )
  }
}