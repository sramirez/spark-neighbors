package com.github.karlhigley.spark.neighbors

import com.github.karlhigley.spark.neighbors.collision.CollisionStrategy
import com.github.karlhigley.spark.neighbors.linalg.DistanceMeasure
import com.github.karlhigley.spark.neighbors.lsh.{HashTableEntry, LSHFunction}
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.ml.feature.LabeledPoint

/**
 * Model containing hash tables produced by computing signatures
 * for each supplied vector.
 */
class ANNModel(val hashTables: RDD[_ <: HashTableEntry[_]],
               val hashFunctions: Iterable[_ <: LSHFunction[_]],
               val collisionStrategy: CollisionStrategy,
               val distance: DistanceMeasure,
               val numPoints: Int) extends Serializable {

  import ANNModel._

  lazy val candidates =
    collisionStrategy
      .apply(hashTables)
      .groupByKey(hashTables.getNumPartitions)
      .values

  /**
   * Identify pairs of nearest neighbors by applying a
   * collision strategy to the hash tables and then computing
   * the actual distance between candidate pairs.
   */
  def neighbors(quantity: Int): RDD[(Long, Array[(Long, Double)])] =
    computeDistances(candidates).topByKey(quantity)(ANNModel.ordering)

  /**
   * Identify the nearest neighbors of a collection of new points
   * by computing their signatures, filtering the hash tables to
   * only potential matches, cogrouping the two RDDs, and
   * computing candidate distances in the "normal" fashion.
   */
  def neighbors(queryPoints: RDD[IDPoint], quantity: Int): RDD[(Long, Array[(Long, Double)])] = {
    val modelEntries = collisionStrategy.apply(hashTables)

    val queryHashTables = ANNModel.generateHashTables(queryPoints, hashFunctions)
    val queryEntries = collisionStrategy.apply(queryHashTables)
    
    val tmp2 = queryEntries.collect()

    val candidateGroups =
      queryEntries.cogroup(modelEntries)
        .values

    val neighbors = computeBipartiteDistances(candidateGroups)

    val tmp = neighbors.take(10)
    neighbors.topByKey(quantity)(ANNModel.ordering)
  }

  /**
   * Compute the average selectivity of the points in the
   * dataset. (See "Modeling LSH for Performance Tuning" in CIKM '08.)
   */
  def avgSelectivity(): Double =
    candidates
      .flatMap {
        case candidates => {
          for (
            (id1, _) <- candidates.iterator;
            (id2, _) <- candidates.iterator
          ) yield (id1, id2)
        }
      }
      .distinct
      .countByKey()
      .values
      .map(_.toDouble / numPoints).reduce(_ + _) / numPoints

  /**
   * Compute the actual distance between candidate pairs using the supplied distance measure.
   */
  private def computeDistances(candidates: RDD[CandidateGroup]): RDD[(Long, (Long, Double))] =
    candidates
      .flatMap {
        case group => {
          for (
            (id1, vector1) <- group.iterator;
            (id2, vector2) <- group.iterator;
            if id1 < id2
          ) yield ((id1, id2), distance(vector1.features, vector2.features))
        }
      }
      .reduceByKey((a, b) => a)
      .flatMap {
        case ((id1, id2), dist) => Seq((id1, (id2, dist)), (id2, (id1, dist)))
      }

  /**
   * Compute the actual distance between candidate pairs
   * using the supplied distance measure.
   */
  private def computeBipartiteDistances(candidates: RDD[(CandidateGroup, CandidateGroup)]): RDD[(Long, (Long, Double))] = {
    candidates
      .flatMap {
        case (groupA, groupB) => {
          for (
            (id1, vector1) <- groupA.iterator;
            (id2, vector2) <- groupB.iterator
            if id1 != id2
          ) yield ((id1, id2), distance(vector1.features, vector2.features))
        }
      }
      .reduceByKey((a, b) => a)
      .map {
        case ((id1, id2), dist) => (id1, (id2, dist))
      }
  }

}

object ANNModel {

  type IDPoint = (Long, LabeledPoint)
  type CandidateGroup = Iterable[IDPoint]

  val ordering = Ordering[Double].on[(Long, Double)](_._2).reverse

  /**
   * Train a model by computing signatures for the supplied points
   */
  def train(points: RDD[IDPoint],
            hashFunctions: Iterable[_ <: LSHFunction[_]],
            collisionStrategy: CollisionStrategy,
            measure: DistanceMeasure,
            persistenceLevel: StorageLevel): ANNModel = {

    val hashTables: RDD[_ <: HashTableEntry[_]] = generateHashTables(points, hashFunctions)

    hashTables.persist(persistenceLevel)

    new ANNModel(
      hashTables,
      hashFunctions,
      collisionStrategy,
      measure,
      points.count.toInt)

  }

  def generateHashTables(points: RDD[(Long, LabeledPoint)],
                         hashFunctions: Iterable[_ <: LSHFunction[_]]): RDD[_ <: HashTableEntry[_]] =
    points
      .flatMap{ case (id, vector) =>
        hashFunctions
          .zipWithIndex
          .map{ case (hashFunc: LSHFunction[_], table: Int) => hashFunc.hashTableEntry(id, table, vector) }}

}