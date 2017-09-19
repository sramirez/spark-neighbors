package com.github.karlhigley.spark.neighbors

import com.github.karlhigley.spark.neighbors.collision.CollisionStrategy
import com.github.karlhigley.spark.neighbors.linalg.DistanceMeasure
import com.github.karlhigley.spark.neighbors.lsh.{HashTableEntry, LSHFunction}
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.broadcast.Broadcast
import com.github.karlhigley.spark.neighbors.ANNModel.IDPoint
import com.github.karlhigley.spark.neighbors.util.{BoundedPriorityQueue => BPQ }

/**
 * Model containing hash tables produced by computing signatures
 * for each supplied vector.
 */
class KnnModel(val points: RDD[IDPoint],
               val distance: DistanceMeasure,
               val numPoints: Int) extends Serializable {

  import KnnModel._
  
  val sc = points.context
  val ordering = Ordering[Float].on[(Float, IDPoint)](_._1).reverse
  /**
   * Identify pairs of nearest neighbors by applying a
   * collision strategy to the hash tables and then computing
   * the actual distance between candidate pairs.
   */
  def neighbors(quantity: Int): RDD[Array[IDPoint]] = {
    val bquery = sc.broadcast(points.collect())
    computeDistances(bquery, quantity)
  }

  /**
   * Identify the nearest neighbors of a collection of new points
   * by computing their signatures, filtering the hash tables to
   * only potential matches, cogrouping the two RDDs, and
   * computing candidate distances in the "normal" fashion.
   */
  def neighbors(queryPoints: RDD[IDPoint], quantity: Int): RDD[Array[IDPoint]] = {
    val bquery = sc.broadcast(queryPoints.collect())    
    computeDistances(bquery, quantity)
  }

  /**
   * Compute the actual distance between candidate pairs using the supplied distance measure.
   */
  private def computeDistances(bquery: Broadcast[Array[IDPoint]], quantity: Int): RDD[Array[IDPoint]] = {
    val neighbors = points.mapPartitions{ it =>
      val query = bquery.value
      val topk = (0 until query.size).map(i => new BPQ[(Float, IDPoint)](quantity)(ordering)).toArray
      while(it.hasNext) {
        val ref = it.next()
        (0 until query.size).map(i => topk(i) += distance(ref._2.features, query(i)._2.features).toFloat -> query(i))
      }
      val oit = (0 until query.size).map(i => query(i)._1 -> topk(i))
      oit.toIterator    
    }.reduceByKey{ case (x, y) => x ++= y}
    neighbors.map(_._2.map(_._2).toArray) 
  }
  
}

object KnnModel {

  /**
   * Train a model by computing signatures for the supplied points
   */
  def train(points: RDD[IDPoint],
            measure: DistanceMeasure,
            persistenceLevel: StorageLevel): KnnModel = {

    new KnnModel(
      points,
      measure,
      points.count.toInt)

  }

}