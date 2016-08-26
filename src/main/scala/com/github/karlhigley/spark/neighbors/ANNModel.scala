package com.github.karlhigley.spark.neighbors

import com.github.karlhigley.spark.neighbors.collision.CollisionStrategy
import com.github.karlhigley.spark.neighbors.linalg.DistanceMeasure
import com.github.karlhigley.spark.neighbors.lsh.{ HashTableEntry, LSHFunction }
import org.apache.spark.mllib.linalg.{ Vector => MLLibVector }
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * Model containing hash tables produced by computing signatures
 * for each supplied vector.
 */
class ANNModel(
  val hashTables: RDD[_ <: HashTableEntry[_]],
    val hashFunctions: Array[_ <: LSHFunction[_]],
    val collisionStrategy: CollisionStrategy,
    val measure: DistanceMeasure,
    val numPoints: Int
) extends Serializable {

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
  def neighbors(queryPoints: RDD[Point], quantity: Int): RDD[(Long, Array[(Long, Double)])] = {
    val modelEntries = collisionStrategy.apply(hashTables)

    val queryHashTables = ANNModel.generateHashTables(queryPoints, hashFunctions)
    val queryEntries = collisionStrategy.apply(queryHashTables)

    val candidateGroups = queryEntries.cogroup(modelEntries).values
    val neighbors = computeBipartiteDistances(candidateGroups)

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
   * Compute the actual distance between candidate pairs
   * using the supplied distance measure.
   */
  private def computeDistances(candidates: RDD[CandidateGroup]): RDD[(Long, (Long, Double))] =
    candidates
      .flatMap {
        case group => {
          for (
            (id1, vector1) <- group.iterator;
            (id2, vector2) <- group.iterator;
            if id1 < id2
          ) yield ((id1, id2), measure.compute(vector1, vector2))
        }
      }
      .reduceByKey((a, b) => a)
      .flatMap {
        case ((id1, id2), dist) => Array((id1, (id2, dist)), (id2, (id1, dist)))
      }

  /**
   * Compute the actual distance between candidate pairs
   * using the supplied distance measure.
   */
  private def computeBipartiteDistances(candidates: RDD[(CandidateGroup, CandidateGroup)]): RDD[(Long, (Long, Double))] =
    candidates
      .flatMap {
        case (groupA, groupB) => {
          for (
            (id1, vector1) <- groupA.iterator;
            (id2, vector2) <- groupB.iterator
          ) yield ((id1, id2), measure.compute(vector1, vector2))
        }
      }
      .reduceByKey((a, b) => a)
      .map {
        case ((id1, id2), dist) => (id1, (id2, dist))
      }

}

object ANNModel {

  type Point = (Long, MLLibVector)
  type CandidateGroup = Iterable[Point]

  val ordering = Ordering[Double].on[(Long, Double)](_._2).reverse

  /**
   * Train a model by computing signatures for the supplied points
   */
  def train(
    points: RDD[(Long, MLLibVector)],
    hashFunctions: Array[_ <: LSHFunction[_]],
    collisionStrategy: CollisionStrategy,
    measure: DistanceMeasure,
    persistenceLevel: StorageLevel
  ): ANNModel = {

    val hashTables: RDD[_ <: HashTableEntry[_]] = generateHashTables(points, hashFunctions)

    hashTables.persist(persistenceLevel)

    new ANNModel(
      hashTables,
      hashFunctions,
      collisionStrategy,
      measure,
      points.count().toInt
    )

  }

  def generateHashTables(
    points: RDD[(Long, MLLibVector)],
    hashFunctions: Array[_ <: LSHFunction[_]]
  ): RDD[_ <: HashTableEntry[_]] =
    points
      .flatMap {
        case (id, vector) =>
          hashFunctions
            .zipWithIndex
            .map { case (hashFunc, table) => hashFunc.hashTableEntry(id, table, vector) }
      }

}