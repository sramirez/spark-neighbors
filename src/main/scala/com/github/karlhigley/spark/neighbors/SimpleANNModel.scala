package com.github.karlhigley.spark.neighbors

import com.github.karlhigley.spark.neighbors.ANNModel.{ CandidateGroup, Point }
import com.github.karlhigley.spark.neighbors.linalg.DistanceMeasure
import com.github.karlhigley.spark.neighbors.lsh.{ HashTableEntry, LSHFunction }
import com.github.karlhigley.spark.neighbors.util.NonSparkHelperFunctions._
import org.apache.spark.mllib.linalg.{ Vector => MLLibVector }

import scala.util.hashing.MurmurHash3

/**
 * A single node counterpart to ANNModel, uses Lists instead of RDDs.
 *
 * @author Thomas Moerman
 */
class SimpleANNModel(val hashTables: Iterable[_ <: HashTableEntry[_]],
                     val hashFunctions: Iterable[_ <: LSHFunction[_]],
                     val distance: DistanceMeasure,
                     val numPoints: Int) extends Serializable {

  import SimpleANNModel._

  lazy val candidates =
    collisionStrategy
      .apply(hashTables)
      .groupBy(_._1)
      .values.map(_.map(_._2))

  /**
   * Identify pairs of nearest neighbors by applying a
   * collision strategy to the hash tables and then computing
   * the actual distance between candidate pairs.
   */
  def neighbors(quantity: Int): Map[Long, Iterable[(Long, Double)]] = {
    val neighbors = computeDistances(candidates)

    topByKey(quantity, neighbors)(ANNModel.ordering)
  }

  /**
   * Identify the nearest neighbors of a collection of new points
   * by computing their signatures, filtering the hash tables to
   * only potential matches, cogrouping the two RDDs, and
   * computing candidate distances in the "normal" fashion.
   */
  def neighbors(queryPoints: Iterable[Point], quantity: Int): Map[Long, Iterable[(Long, Double)]] = {
    val modelEntries = collisionStrategy.apply(hashTables)

    val queryHashTables = SimpleANNModel.generateHashTables(queryPoints, hashFunctions)
    val queryEntries = collisionStrategy.apply(queryHashTables)

    val candidateGroups = cogroup(queryEntries, modelEntries).values
    val neighbors = computeBipartiteDistances(candidateGroups)

    topByKey(quantity, neighbors)(ANNModel.ordering)
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
      .toSet
      .foldLeft(Map[Long, Int]().withDefaultValue(0)) { case (acc, (id1, _)) => acc.updated(id1, acc(id1) + 1) }
      .values
      .map(_.toDouble / numPoints).reduce(_ + _) / numPoints

  /**
   * Compute the actual distance between candidate pairs using the supplied distance measure.
   */
  private def computeDistances(candidates: Iterable[CandidateGroup]): Iterable[(Long, (Long, Double))] =
    candidates
      .flatMap {
        case group =>
          for {
            (id1, vector1) <- group.iterator
            (id2, vector2) <- group.iterator
            if id1 < id2
          } yield ((id1, id2), distance(vector1, vector2))
      }
      .foldLeft(Map[(Long, Long), Double]()) {
        case (acc, (ids, distance)) =>
          acc + (ids -> distance)
      }
      .flatMap {
        case ((id1, id2), distance) =>
          Seq(
            (id1, (id2, distance)),
            (id2, (id1, distance))
          )
      }

  /**
   * Compute the actual distance between candidate pairs using the supplied distance measure.
   */
  private def computeBipartiteDistances(candidates: Iterable[(CandidateGroup, CandidateGroup)]): Iterable[(Long, (Long, Double))] =
    candidates
      .flatMap {
        case (groupA, groupB) =>
          for {
            (id1, vector1) <- groupA.iterator
            (id2, vector2) <- groupB.iterator
          } yield ((id1, id2), distance(vector1, vector2))
      }
      .foldLeft(Map[(Long, Long), Double]()) {
        case (acc, (ids, distance)) =>
          acc + (ids -> distance)
      }
      .map {
        case ((id1, id2), dist) =>
          (id1, (id2, dist))
      }

}

object SimpleANNModel {

  /**
   * Train a model by computing signatures for the supplied
   * points
   */
  def train(points: Iterable[(Long, MLLibVector)],
            hashFunctions: Iterable[_ <: LSHFunction[_]],
            measure: DistanceMeasure) =
    new SimpleANNModel(
      generateHashTables(points, hashFunctions),
      hashFunctions,
      measure,
      points.size)

  def generateHashTables(points: Iterable[(Long, MLLibVector)],
                         hashFunctions: Iterable[_ <: LSHFunction[_]]): Iterable[_ <: HashTableEntry[_]] =
    points
      .flatMap {
        case (id, vector) =>
          hashFunctions
            .zipWithIndex
            .map { case (hashFunc: LSHFunction[_], table: Int) => hashFunc.hashTableEntry(id, table, vector) }}

  val collisionStrategy = (hashTables: Iterable[_ <: HashTableEntry[_]]) =>
    hashTables
      .map(entry => {
        val key = (entry.table, MurmurHash3.arrayHash(entry.sigElements)).asInstanceOf[Product]
        (key, (entry.id, entry.point)) })

}