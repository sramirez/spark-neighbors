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
 * Exact Nearest Neighbors (KNN) using broadcasting of query instances
 *
 * @see J. Maillo, S. Ramírez-Gallego, I. Triguero, F. Herrera. 
 * kNN-IS: An Iterative Spark-based design of the k-Nearest Neighbors classifier for big data. 
 * Knowledge-Based Systems, in press. [doi: 10.1016/j.knosys.2016.06.012]
 */

class KNN private[neighbors] (private val measureName: String = "euclidean") {


  /**
   * Build an KNN model using the given dataset.
   *
   * @param points    RDD of vectors paired with IDs.
   *                   IDs must be unique and >= 0.
   * @return KNNModel containing computed hash tables
   */
  def train(points: RDD[IDPoint],
            persistenceLevel: StorageLevel = MEMORY_AND_DISK): KnnModel = {

    val distanceMeasure = measureName.toLowerCase match {
      case "hamming" => HammingDistance
      case "cosine" => CosineDistance
      case "euclidean" => EuclideanDistance
      case "manhattan" => ManhattanDistance
      case "fractional" => FractionalDistance
      case "jaccard" => JaccardDistance
      case other: Any =>
        throw new IllegalArgumentException(
          s"Only hamming, cosine, euclidean, manhattan, and jaccard distances are supported but got $other."
        )

    }

    KnnModel.train(
      points,
      distanceMeasure,
      persistenceLevel
    )
  }

}