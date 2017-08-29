package com.github.karlhigley.spark.neighbors

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import com.github.karlhigley.spark.neighbors.collision.CollisionStrategy
import com.github.karlhigley.spark.neighbors.linalg.DistanceMeasure
import com.github.karlhigley.spark.neighbors.lsh.{HashTableEntry, LSHFunction}
import org.apache.spark.mllib.linalg.{Vector => MLLibVector}
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.Vectors
import com.github.karlhigley.spark.neighbors.ANNModel.IDPoint
import org.apache.hadoop.mapreduce.Partitioner
import com.github.karlhigley.spark.neighbors.lsh.SignRandomProjectionFunction
import org.apache.spark.ml.feature.LabeledPoint
import com.github.karlhigley.spark.neighbors.lsh.BitSignature
import com.github.karlhigley.spark.neighbors.linalg.BitHammingDistance
import com.github.karlhigley.spark.neighbors.lsh.BitHashTableEntry
import com.github.karlhigley.spark.neighbors.lsh.BitHashTableEntry

/**
 * Model containing hash tables produced by computing signatures
 * for each supplied vector.
 */
class fastANNModel(val entries: RDD[(Double, _ <: HashTableEntry[_])],
               val hashFunctions: Iterable[_ <: LSHFunction[_]],
               val collisionStrategy: CollisionStrategy,
               val distance: DistanceMeasure,
               val numPoints: Int,
               val signatureLength: Int,
               val thLength: Int = 5,
               val thDistance: Float = .9f) extends Serializable {

  import fastANNModel._

  val ordering = new Ordering[Tuple2[Double, _ <: HashTableEntry[_]]]() {
    override def compare(x: (Double, _), y: (Double, _)): Int = 
        Ordering[Double].compare(x._1, y._1)
    }
  val normMax = entries.max()(ordering)._1
  val normMin = entries.min()(ordering)._1
  
  val r = (normMax - normMin) / thLength
  val tau = signatureLength * math.acos(thDistance).toFloat
  val factor = math.Pi / signatureLength
  
  //val pad = entries.glom().map(a => (a.head._1 - r, a.last._1 + r))
  //val bcPads = entries.context.broadcast(pad)
  
  
  /*lazy val candidates =
    collisionStrategy
      .apply(entries)
      .groupByKey(entries.getNumPartitions)
      .values */

  /**
   * Identify pairs of nearest neighbors by applying a
   * collision strategy to the hash tables and then computing
   * the actual distance between candidate pairs.
   */
  def neighbors(quantity: Int): RDD[(Long, Array[BitHashTableEntry])] = {
    
    entries.mapPartitionsWithIndex{ (index, iterator) => {
          val elems = iterator.toSeq
          val output = (0 until elems.size).map{ i =>
            var tree = scala.collection.immutable.TreeMap.empty[Float, BitHashTableEntry]
            val hi = elems(i)._2.asInstanceOf[BitHashTableEntry]
            val ni = elems(i)._2.norm
            
            def loopNeighbors(j: Int) = {
              val hj = elems(j)._2.asInstanceOf[BitHashTableEntry]
              val hamdist = BitHammingDistance.apply(hi.signature, hj.signature)
              if(hamdist <= tau){
                val nj = elems(j)._2.norm
                val eudist = math.pow(ni, 2) + math.pow(nj, 2) -
                  2 * ni * nj * math.cos(factor * hamdist)
                
                tree += eudist.toFloat -> hj
              }
            } 
            
            // traverse from here to left (till r radius limit)    
            for(j <- i until 0 if elems(i)._1 - elems(j)._1 <= r)  
              loopNeighbors(j)  
             
            
            // traverse from here to right
            for(j <- i until elems.size if elems(j)._1 - elems(i)._1 <= r)
              loopNeighbors(j)  
            
            (hi.id, tree.takeRight(quantity).map(_._2).toArray)
          }
          output.iterator
        }
    }
  }

  /**
   * Identify the nearest neighbors of a collection of new points
   * by computing their signatures, filtering the hash tables to
   * only potential matches, cogrouping the two RDDs, and
   * computing candidate distances in the "normal" fashion.
   */
  def neighbors(queryPoints: RDD[IDPoint], quantity: Int): RDD[(Long, Array[(Long, Double)])] = {

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
  private def computeBipartiteDistances(candidates: RDD[(CandidateGroup, CandidateGroup)]): RDD[(Long, (Long, Double))] =
    candidates
      .flatMap {
        case (groupA, groupB) => {
          for (
            (id1, vector1) <- groupA.iterator;
            (id2, vector2) <- groupB.iterator
          ) yield ((id1, id2), distance(vector1.features, vector2.features))
        }
      }
      .reduceByKey((a, b) => a)
      .map {
        case ((id1, id2), dist) => (id1, (id2, dist))
      }

}


object fastANNModel {

  type CandidateGroup = Iterable[IDPoint]

  /**
   * Train a model by computing signatures for the supplied points
   */
  def train(points: RDD[IDPoint],
            hashFunctions: Iterable[_ <: LSHFunction[_]],
            collisionStrategy: CollisionStrategy,
            measure: DistanceMeasure,
            signatureLength: Int,
            persistenceLevel: StorageLevel): fastANNModel = {

    val hashTables = generateHashTables(points, hashFunctions).sortByKey()
    
    hashTables.persist(persistenceLevel)

    new fastANNModel(
      hashTables,
      hashFunctions,
      collisionStrategy,
      measure,
      points.count.toInt,
      signatureLength)

  }

  def generateHashTables(points: RDD[IDPoint],
                         hashFunctions: Iterable[_ <: LSHFunction[_]]): RDD[(Double, HashTableEntry[_])] =
    points
      .flatMap{ case (id, vector) =>
        hashFunctions
          .zipWithIndex
          .map{ case (hashFunc: LSHFunction[_], table: Int) => 
            val entry = hashFunc.hashTableEntry(id, table, vector)
            entry.norm -> entry
          }
        }

}