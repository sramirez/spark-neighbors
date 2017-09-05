package com.github.karlhigley.spark.neighbors

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.hadoop.mapreduce.Partitioner
import com.github.karlhigley.spark.neighbors.collision.CollisionStrategy
import com.github.karlhigley.spark.neighbors.linalg.DistanceMeasure
import com.github.karlhigley.spark.neighbors.lsh.{HashTableEntry, LSHFunction}
import com.github.karlhigley.spark.neighbors.lsh.BitSignature
import com.github.karlhigley.spark.neighbors.lsh.{BitHashTableEntry => BHTE}
import com.github.karlhigley.spark.neighbors.linalg.BitHammingDistance
import com.github.karlhigley.spark.neighbors.util.{BoundedPriorityQueue => BPQ }
import com.github.karlhigley.spark.neighbors.ANNModel.IDPoint
import scala.collection.Searching._
import com.github.karlhigley.spark.neighbors.lsh.BitHashTableEntry
import com.github.karlhigley.spark.neighbors.linalg.EuclideanDistance

/**
 * Model containing hash tables produced by computing signatures
 * for each supplied vector.
 */
class fastANNModel(var entries: RDD[(Float, BHTE)],
               val hashFunctions: Iterable[_ <: LSHFunction[_]],
               val collisionStrategy: CollisionStrategy,
               val distance: DistanceMeasure,
               val numPoints: Int,
               val signatureLength: Int,               
               val nClasses: Int,
               val persistenceLevel: StorageLevel,
               val thLength: Int = 5,
               val thDistance: Float = .9f) extends Serializable {

  import fastANNModel._

  
  val ordering = Ordering[Float].on[(Float, BHTE)](_._1).reverse
  
  /* Variables needed for further estimation of distance */
  val normMax = entries.max()(ordering.reverse)._1
  val normMin = entries.min()(ordering.reverse)._1
  
  val r = (normMax - normMin) / thLength
  val tau = signatureLength * math.acos(thDistance).toFloat
  // pi radians / |m|
  val weightHamming = math.Pi / signatureLength
  
  /* It indicates if fuzzy memberships are computed, and then fuzzy prediciton can be used. */
  var fuzzy = false  
  
    
  /** 
   *  Function that computes approximate euclidean distance from two elements
   *  represented by two bit encoding vectors. Hamming distance between these two 
   *  bit vectors is used to approximate the euclidean one.
   *  
   *  Vectors with a hamming distance far from tau threshold are omitted.
   */
  private[neighbors] def approxEuclidDistance(a: BHTE, b: BHTE): Option[(Float, BHTE)] = {
    
    val hamdist = BitHammingDistance.apply(a.signature, b.signature)
    //val realDistance = EuclideanDistance.apply(a.point.features, b.point.features)
    if(hamdist <= tau){
      val ni = a.norm; val nj = b.norm
      
      val eudist = math.sqrt(math.pow(ni, 2) + math.pow(nj, 2) -
        2 * ni * nj * math.cos(weightHamming * hamdist))    
      //val error = math.abs(realDistance - eudist) / realDistance
      //val estcosi = math.cos(weightHamming * hamdist)
      //val cos = -(math.pow(realDistance, 2) - math.pow(ni, 2) - math.pow(nj, 2)) / (2 * ni * nj)
      val approxDist = if(!java.lang.Double.isNaN(eudist)) eudist.toFloat else .0f
      return Some(approxDist -> b)
    }
    return None
  }
      
  /**
   * Fast search algorithm following the paper (2013, Marukatat).
   * This function computes approximate euclidean distance from two elements
   * represented by two bit encoding vectors, skipping those vectors whose 
   * distance is far from the radius limit (leftIndex, rightIndex) and the encoder length.
   * 
   * Pre-requisites: elems must be sorted.
   */
  private def fastNearestSearch(quantity: Int, 
      elems: Seq[(Float, BHTE)], // must be sorted
      leftIndex: Int, 
      rightIndex: Int,
      // Norm, HashEntry
      q: (Float, BitHashTableEntry)): BPQ[(Float, BHTE)] = {
    
    
    import scala.util.control.Breaks._
    /* k Nearest neighbors ordered by euclidean distance */
    var topk = new BPQ[(Float, BHTE)](quantity)(ordering)
    //println("elems: " + elems.map(_._2.id).toString())
    
    // traverse from here to left (till r radius limit)    
    breakable{ 
      for(j <- leftIndex to 0 by -1){
        /*if(elems(j)._2.id == q._2.id){
          println("ID query: "  + q._2.id)
          println("Index query: " + index)
          println("ID : "  + elems(j)._2.id)
          println("Index: " + j)
          
          println("elems: " + elems.map(_._2.id).toString())
        }*/
        //if(q._1 - elems(j)._1 > r)
          //break
        approxEuclidDistance(q._2, elems(j)._2) match {
          case Some(distNeig) => topk += distNeig
          case None => /* Skip this example. */
        }
      }
    }
  
    // traverse from here to right
    breakable{ 
      for(j <- rightIndex until elems.size){
        //if(elems(j)._1 - q._1 > r)
          //break  
        approxEuclidDistance(q._2, elems(j)._2) match {
          case Some(distNeig) => topk += distNeig
          case None => /* Skip this example. */
        }
      }
    }
    
    topk
  } 

  /**
   * Apply fast nearest neighbor search to each element in entries. 
   * Searches are applied locally (in each data partition). Some errors
   * in searches are assumed in the partition limits.
   */
  private[neighbors] def fullNeighbors(quantity: Int): RDD[(BHTE, Seq[(Float, BHTE)])] = {
    
    entries.mapPartitions{ iterator => 
        val elems = iterator.toArray
        val output = (0 until elems.size).map{ i =>
          val topk = fastNearestSearch(quantity, elems, i - 1, i + 1, elems(i))
          (elems(i)._2, topk.toSeq)
        }
        output.iterator
    }
  }
  
  /**
   * Compute fuzzy membership values for all elements in the case-base.
   * It relies on fullNeighbors function to compute distance between the neighbors.
   * It followed the description of fuzzy kNN stated in (derrac, 2014).
   * 
   * Retrieved entries will replace old hash entries. They are sorted and persisted 
   * for further searches.
   */
  private def computeFuzzyMembership(k: Int, nClasses: Int) {
    
    val input = fullNeighbors(k)
    entries = input.map{ case (orig, neighbors) =>
      var counter = Array.fill[Float](nClasses)(.0f)
      neighbors.map{ case (_, neig) => counter(neig.point.label.toInt) += 1 }
      counter = counter.map(_ / k * .49f)
      counter(orig.point.label.toInt) += .51f
      // we get the first two decimals to represent membership
      orig.fuzzyMembership = counter.map(num => math.floor(num * 100).toByte)
      (orig.norm, orig)
    }.sortByKey().persist(persistenceLevel)
    
    fuzzy = true
    
  }
    
  /**
   * Compute the fuzzy prediction for a given instance following the standard fuzzy knn rules.
   * 
   */
  private def fuzzyPrediction(topNeighbors: BPQ[(Float, BHTE)], m: Int): Int = {
      
      var fuzzydist = topNeighbors.map{ case (dist, _) => (1 / math.pow(dist, 2 / (m - 1))).toFloat }.toSeq
      val totald = fuzzydist.sum
      fuzzydist = fuzzydist.map(_ / totald) // normalize distance
      val fuzzymemb = topNeighbors.map(_._2).toSeq
          
      val count = Array.fill[Float](nClasses)(0)
      (0 until nClasses).map{ cls => 
       (0 until topNeighbors.size).map{ i =>
         // memberships are saved in byte format representing the two first decimals (e.g.: 0.49)
         count(cls) += fuzzymemb(i).fuzzyMembership(cls) / 100.0f * fuzzydist(i)
       }
      }
      
      /* Compute the maximum fuzzy contribution */
      var pred = 0
      var max = Float.MinValue
      for(i <- 0 until count.size) {
        if(count(i) > max){
          pred = i
          max = count(i)
        }                    
      }
      pred
  }
  
  /**
   * Wrapper function to retrieve the nearest neighbors for all elements already 
   * stored in the case-base. It follows the same scheme proposed in the ANNmodel class.
   */
  def neighbors(quantity: Int): RDD[(Long, Array[(Long, Double)])] = {
    fullNeighbors(quantity).map{ case (o, seq) => 
      o.id -> seq.map{ case(dist, e) => e.id -> dist.toDouble}.toArray
    }
  }    
   
  /**
   * Wrapper function to retrieve the nearest neighbors to a bunch of new points.
   * It follows the same scheme proposed in the ANNmodel class.
   */
  def neighbors(queryPoints: RDD[IDPoint], quantity: Int): RDD[(Long, Array[(Long, Double)])] = {
    val queryEntries = fastANNModel
      .generateHashTables(queryPoints, hashFunctions)
      .sortByKey(numPartitions = entries.getNumPartitions)      
      
    // for each partition, search points within corresponding child tree
    entries.zipPartitions(queryEntries, preservesPartitioning = true) {
      (itEntries, itQuery) =>
        val elems = itEntries.toIndexedSeq
        itQuery.map { q =>            
            val i = elems.search(q)(ordering).insertionPoint
            val topNeighbors = fastNearestSearch(quantity, elems, i - 1, i, q)
              .map{ case(dist, e) => (e.id, dist.toDouble) }.toArray
            q._2.id -> topNeighbors
        }
    }
  }

  /**
   * Compute kNN prediction in two modes: majority crisp and fuzzy voting.
   * Query points are hashed and sorted by norm. Then, old and query points are
   * zipped together by partitions, and searches are performed. 
   */
  def predict(queryPoints: RDD[IDPoint], 
      quantity: Int, 
      mFuzzy: Option[Int] = Some(2)): RDD[(Long, (Double, Double))] = {
    
    /* Fuzzy memberships are pre-computed. If kFuzzy was not specified, fuzzy prediction is not valid. */
    assert(mFuzzy.isDefined && fuzzy)
    
    val queryEntries = fastANNModel
      .generateHashTables(queryPoints, hashFunctions)
      .sortByKey(numPartitions = entries.getNumPartitions)
      
    // for each partition, search points within corresponding child tree
    entries.zipPartitions(queryEntries, preservesPartitioning = true) {
      (itEntries, itQuery) =>
        val elems = itEntries.toIndexedSeq
        itQuery.map { q =>
            val i = elems.search(q)(ordering).insertionPoint
            val topNeighbors = fastNearestSearch(quantity, elems, i - 1, i, q)
            val actual = q._2.point.label
            mFuzzy match {              
              case Some(m) =>
                val pred = fuzzyPrediction(topNeighbors, m)
                (q._2.id, (pred, actual))
              
              case None =>
                val pred = topNeighbors.map(_._2.point.label)
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
            persistenceLevel: StorageLevel,
            kFuzzy: Int = 0): fastANNModel = {

    val hashTables = generateHashTables(points, hashFunctions).sortByKey()
    hashTables.persist(persistenceLevel)

    val model = new fastANNModel(
      hashTables,
      hashFunctions,
      collisionStrategy,
      measure,
      points.count.toInt,
      signatureLength,
      nClasses,
      persistenceLevel)
    
    if(kFuzzy > 0) 
      model.computeFuzzyMembership(kFuzzy, nClasses)
      
    model

  }

  def generateHashTables(points: RDD[IDPoint],
                         hashFunctions: Iterable[_ <: LSHFunction[_]]): RDD[(Float, BHTE)] =
    points
      .flatMap{ case (id, vector) =>
        hashFunctions
          .zipWithIndex
          .map{ case (hashFunc: LSHFunction[_], table: Int) => 
            val entry = hashFunc.hashTableEntry(id, table, vector)
            entry.sigElements.size.toFloat -> entry.asInstanceOf[BHTE]
            //entry.sigElements.size.toFloat -> entry.asInstanceOf[BHTE]
          }
        }

}