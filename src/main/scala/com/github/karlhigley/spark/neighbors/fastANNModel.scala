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
import com.github.karlhigley.spark.neighbors.util.BoundedPriorityQueue
import scala.collection.Searching._

/**
 * Model containing hash tables produced by computing signatures
 * for each supplied vector.
 */
class fastANNModel(val entries: RDD[(Float, BitHashTableEntry)],
               val hashFunctions: Iterable[_ <: LSHFunction[_]],
               val collisionStrategy: CollisionStrategy,
               val distance: DistanceMeasure,
               val numPoints: Int,
               val signatureLength: Int,               
               val nClasses: Int,
               val thLength: Int = 5,
               val thDistance: Float = .9f) extends Serializable {

  import fastANNModel._

  val ordering = new Ordering[Tuple2[Float, BitHashTableEntry]]() {
    override def compare(x: (Float, BitHashTableEntry), y: (Float, BitHashTableEntry)): Int = 
        Ordering[Float].compare(x._1, y._1)
  }
  val normMax = entries.max()(ordering)._1
  val normMin = entries.min()(ordering)._1
  
  val r = (normMax - normMin) / thLength
  val tau = signatureLength * math.acos(thDistance).toFloat
  val factor = math.Pi / signatureLength
  
  
    
  private def approxEuclidDistance(a: BitHashTableEntry, 
      b: BitHashTableEntry): Option[(Float, BitHashTableEntry)] = {
    
    val hamdist = BitHammingDistance.apply(a.signature, b.signature)
    if(hamdist <= tau){
      val ni = a.norm; val nj = b.norm
      val eudist = math.pow(ni, 2) + math.pow(nj, 2) -
        2 * ni * nj * math.cos(factor * hamdist)                
      Some(eudist.toFloat -> b)
    }
    None
  }
      
  private def fastNearestSearch(quantity: Int, 
      elems: Seq[(Float, BitHashTableEntry)], // must be sorted
      leftIndex: Int, 
      rightIndex: Int,
      index: Int): BoundedPriorityQueue[(Float, BitHashTableEntry)] = {
    
    assert(rightIndex >= index || leftIndex <= index)
    
    import scala.util.control.Breaks._
    /* k Nearest neighbors ordered by euclidean distance */
    var topk = new BoundedPriorityQueue[(Float, BitHashTableEntry)](quantity)(ordering)
    
    // traverse from here to left (till r radius limit)    
    breakable{ 
      for(j <- leftIndex until 0){
        if(elems(index)._1 - elems(j)._1 > r)
          break
        approxEuclidDistance(elems(index)._2, elems(j)._2) match {
          case Some(distNeig) => topk += distNeig
          case None => /* Skip this example. */
        }
      }
    }
  
    // traverse from here to right
    breakable{ 
      for(j <- rightIndex until elems.size){
        if(elems(j)._1 - elems(index)._1 > r)
          break
        approxEuclidDistance(elems(index)._2, elems(j)._2) match {
          case Some(distNeig) => topk += distNeig
          case None => /* Skip this example. */
        }
      }
    }
    
    topk
  } 

  /**
   * Identify pairs of nearest neighbors by applying a
   * collision strategy to the hash tables and then computing
   * the actual distance between candidate pairs.
   */
  private def fullNeighbors(quantity: Int): RDD[(BitHashTableEntry, Seq[(Float, BitHashTableEntry)])] = {
    
    entries.mapPartitions{ iterator => 
        val elems = iterator.toSeq
        val output = (0 until elems.size).map{ i =>
          val topk = fastNearestSearch(quantity, elems, i - 1, i + 1, i)
          (elems(i)._2, topk.toSeq)
        }
        output.iterator
    }
  }
  
  def computeFuzzyMembership(k: Int, nClasses: Int): RDD[(Float, BitHashTableEntry)] = {
    
    val input = fullNeighbors(k)
    input.map{ case (orig, neighbors) =>
      var counter = Array.fill[Float](nClasses)(.0f)
      neighbors.map{ case (_, neig) => counter(neig.point.label.toInt) += 1 }
      counter = counter.map(_ / k * .49f)
      counter(orig.point.label.toInt) += .51f
      // we get the first two decimals to represent membership
      orig.fuzzyMembership = counter.map(num => math.floor(num * 100).toByte)
      (orig.norm, orig)
    }
  }

  /**
   * Identify the nearest neighbors of a collection of new points
   * by computing their signatures, filtering the hash tables to
   * only potential matches, cogrouping the two RDDs, and
   * computing candidate distances in the "normal" fashion.
   */
  def neighborPrediction(queryPoints: RDD[IDPoint], 
      quantity: Int, 
      mFuzzy: Option[Int] = Some(2)): RDD[(Long, (Double, Double))] = {
    
    val queryEntries = fastANNModel
      .generateHashTables(queryPoints, hashFunctions)
      .sortByKey(numPartitions = entries.getNumPartitions)
      
    // for each partition, search points within corresponding child tree
    entries.zipPartitions(queryEntries, preservesPartitioning = true) {
      (itEntries, itQuery) =>
        val elems = itEntries.toIndexedSeq
        itQuery.map { q =>
            val i = elems.search(q)(ordering).insertionPoint
            val topk = fastNearestSearch(quantity, elems, i - 1, i, i)
            val actual = q._2.point.label
            mFuzzy match {
              
              case Some(m) =>
                var fuzzydist = topk.map{ case (dist, _) => (1 / math.pow(dist, 2 / (m - 1))).toFloat }.toSeq
                val totald = fuzzydist.sum
                fuzzydist = fuzzydist.map(_ / totald)
                val fuzzymemb = topk.map(_._2).toSeq
                
                val count = Array.fill[Float](nClasses)(0)
                (0 until nClasses).map{ cls => 
                 (0 until topk.size).map{ i =>
                   count(cls) += fuzzymemb(i).fuzzyMembership(cls) * fuzzydist(i)
                 }
                }
                
                /** Compute the maximum fuzzy contribution **/
                var pred = 0
                var max = Float.MinValue
                for(i <- 0 until count.size) {
                  if(count(i) > max){
                    pred = i
                    max = count(i)
                  }                    
                }
                
                (q._2.id, (pred, actual))
              
              case None =>
                val pred = topk.map(_._2.point.label)
                  .groupBy(identity).mapValues(_.size)
                  .maxBy(_._2)._1
                (q._2.id, (pred, actual))
            }
        }        
    }
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
            nClasses: Int,
            persistenceLevel: StorageLevel): fastANNModel = {

    val hashTables = generateHashTables(points, hashFunctions).sortByKey()
    
    hashTables.persist(persistenceLevel)

    new fastANNModel(
      hashTables,
      hashFunctions,
      collisionStrategy,
      measure,
      points.count.toInt,
      signatureLength,
      nClasses)

  }

  def generateHashTables(points: RDD[IDPoint],
                         hashFunctions: Iterable[_ <: LSHFunction[_]]): RDD[(Float, BitHashTableEntry)] =
    points
      .flatMap{ case (id, vector) =>
        hashFunctions
          .zipWithIndex
          .map{ case (hashFunc: LSHFunction[_], table: Int) => 
            val entry = hashFunc.hashTableEntry(id, table, vector)
            entry.norm -> entry.asInstanceOf[BitHashTableEntry]
          }
        }

}