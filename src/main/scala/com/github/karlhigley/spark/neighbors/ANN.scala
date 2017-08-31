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
class ANN private (
    private var measureName: String,
    private var origDimension: Int,
    private var numTables: Int,
    private var signatureLength: Int,
    private var bucketWidth: Double,
    private var primeModulus: Int,
    private var numBands: Int,
    private var randomSeed: Int
) {

  /**
   * Constructs an ANN instance with default parameters.
   */
  def this(dimensions: Int, measure: String) = {
    this(
      origDimension = dimensions,
      measureName = measure,
      numTables = 1,
      signatureLength = 16,
      bucketWidth = 0.0,
      primeModulus = 0,
      numBands = 0,
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
   * Bucket width (commonly named "W") used by scalar-random-projection hash functions.
   */
  def getBucketWidth(): Double = {
    bucketWidth
  }

  /**
   * Bucket width (commonly named "W") used by scalar-random-projection hash functions.
   */
  def setBucketWidth(width: Double): this.type = {
    require(
      measureName == "euclidean" || measureName == "manhattan" || measureName == "fractional",
      "Bucket width only applies when distance measure is euclidean or manhattan."
    )
    bucketWidth = width
    this
  }

  /**
   * Common prime modulus used by minhash functions.
   */
  def getPrimeModulus(): Int = {
    primeModulus
  }

  /**
   * Common prime modulus used by minhash functions.
   *
   * Should be larger than the number of dimensions.
   */
  def setPrimeModulus(prime: Int): this.type = {
    require(
      measureName == "jaccard",
      "Prime modulus only applies when distance measure is jaccard."
    )
    primeModulus = prime
    this
  }

  /**
   * Number of bands to use for minhash candidate pair generation
   */
  def getBands(): Int = {
    numBands
  }

  /**
   * Number of bands to use for minhash candidate pair generation
   */
  def setBands(bands: Int): this.type = {
    require(
      measureName == "jaccard",
      "Number of bands only applies when distance measure is jaccard."
    )
    numBands = bands
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
   * Build an ANN model using the given dataset.
   *
   * @param points    RDD of vectors paired with IDs.
   *                   IDs must be unique and >= 0.
   * @return ANNModel containing computed hash tables
   */
  def train(points: RDD[IDPoint],
            persistenceLevel: StorageLevel = MEMORY_AND_DISK): ANNModel = {

    val random = new JavaRandom(randomSeed)

    val (distanceMeasure, hashFunctions, candidateStrategy) = measureName.toLowerCase match {

      case "hamming" => {
        val hashFunctions: Seq[LSHFunction[_]] = (1 to numTables).map(i => BitSamplingFunction.generate(origDimension, signatureLength, random))

        (HammingDistance, hashFunctions, SimpleCollisionStrategy)
      }

      case "cosine" => {
        val functions: Seq[LSHFunction[_]] = (1 to numTables).map(i => SignRandomProjectionFunction.generate(origDimension, signatureLength, random))

        (CosineDistance, functions, SimpleCollisionStrategy)
      }

      case "euclidean" => {
        require(bucketWidth > 0.0, "Bucket width must be greater than zero.")

        val functions: Seq[LSHFunction[_]] = (1 to numTables).map(i => generateL2(origDimension, signatureLength, bucketWidth, random))

        (EuclideanDistance, functions, SimpleCollisionStrategy)
      }

      case "manhattan" => {
        require(bucketWidth > 0.0, "Bucket width must be greater than zero.")

        val functions: Seq[LSHFunction[_]] = (1 to numTables).map(i => generateL1(origDimension, signatureLength, bucketWidth, random))

        (ManhattanDistance, functions, SimpleCollisionStrategy)
      }

      case "fractional" => {
        require(bucketWidth > 0.0, "Bucket width must be greater than zero.")

        val functions: Seq[LSHFunction[_]] = (1 to numTables).map(i => generateFractional(origDimension, signatureLength, bucketWidth, random))

        (FractionalDistance, functions, SimpleCollisionStrategy)
      }

      case "jaccard" => {
        require(primeModulus > 0, "Prime modulus must be greater than zero.")
        require(numBands > 0, "Number of bands must be greater than zero.")
        require(
          signatureLength % numBands == 0,
          "Number of bands must evenly divide signature length."
        )

        val hashFunctions: Seq[LSHFunction[_]] = (1 to numTables).map(i => MinhashFunction.generate(origDimension, signatureLength, primeModulus, random))

        (JaccardDistance, hashFunctions, new BandingCollisionStrategy(numBands))
      }

      case other: Any =>
        throw new IllegalArgumentException(
          s"Only hamming, cosine, euclidean, manhattan, and jaccard distances are supported but got $other."
        )

    }

    ANNModel.train(
      points,
      hashFunctions,
      candidateStrategy,
      distanceMeasure,
      persistenceLevel
    )
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
            thDistance: Float = .9f,
            nClasses: Int,
            persistenceLevel: StorageLevel = MEMORY_AND_DISK): fastANNModel = {

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

  def train(points: Iterable[IDPoint]): SimpleANNModel = {

    val random = new JavaRandom(randomSeed)

    val (distanceMeasure, hashFunctions) = measureName.toLowerCase match {

      case "hamming" => {
        val hashFunctions: Seq[LSHFunction[_]] = (1 to numTables).map(i => BitSamplingFunction.generate(origDimension, signatureLength, random))

        (HammingDistance, hashFunctions)
      }

      case "cosine" => {
        val functions: Seq[LSHFunction[_]] = (1 to numTables).map(i => SignRandomProjectionFunction.generate(origDimension, signatureLength, random))

        (CosineDistance, functions)
      }

      case "euclidean" => {
        require(bucketWidth > 0.0, "Bucket width must be greater than zero.")

        val functions: Seq[LSHFunction[_]] = (1 to numTables).map(i => generateL2(origDimension, signatureLength, bucketWidth, random))

        (EuclideanDistance, functions)
      }

      case "manhattan" => {
        require(bucketWidth > 0.0, "Bucket width must be greater than zero.")

        val functions: Seq[LSHFunction[_]] = (1 to numTables).map(i => generateL1(origDimension, signatureLength, bucketWidth, random))

        (ManhattanDistance, functions)
      }

      case "fractional" => {
        require(bucketWidth > 0.0, "Bucket width must be greater than zero.")

        val functions: Seq[LSHFunction[_]] = (1 to numTables).map(i => generateFractional(origDimension, signatureLength, bucketWidth, random))

        (FractionalDistance, functions)
      }

      case other: Any =>
        throw new IllegalArgumentException(
          s"Only hamming, cosine, euclidean, and manhattan distances are supported but got $other."
        )

    }

    SimpleANNModel.train(points, hashFunctions, distanceMeasure)
  }

}