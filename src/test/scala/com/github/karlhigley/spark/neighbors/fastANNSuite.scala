package com.github.karlhigley.spark.neighbors

import org.scalatest.FunSuite
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.{ Vector => MLLibVector }
import com.github.karlhigley.spark.neighbors.lsh.HashTableEntry
import org.apache.spark.ml.feature.LabeledPoint
import com.github.karlhigley.spark.neighbors.ANNModel.IDPoint
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import com.github.karlhigley.spark.neighbors.util.BoundedPriorityQueue
import com.github.karlhigley.spark.neighbors.ANNModel.IDPoint
import com.github.karlhigley.spark.neighbors.linalg.EuclideanDistance
import org.apache.spark.ml.linalg.Vectors

@RunWith(classOf[JUnitRunner])
class fastANNSuite extends FunSuite with TestSparkContext {
  import fastANNSuite._

  val numPoints = 1000
  val dimensions = 100
  val density = 0.1
  val nClasses = 3

  var sparsePoints: RDD[IDPoint] = _
  var densePoints:  RDD[IDPoint] = _

  override def beforeAll() {
    super.beforeAll()
    val localPoints = TestHelpers.generateRandomLabeledPoint(numPoints, dimensions, density, nClasses)
    sparsePoints = sc.parallelize(localPoints).zipWithIndex.map(_.swap)
    println("Number of partitions generated: " + sparsePoints.partitions.size)
    densePoints = sparsePoints.mapValues(slp => new LabeledPoint(slp.label, slp.features.toDense))
  }

  test("compare fast nearest neighbor search with standard local search") {
    
    val k = 5
    val fastAnn =
      new fastANN(dimensions)
        .setRandomSeed(1234567)
    
    val fastModel = fastAnn.fastANNtrain(sparsePoints, nClasses)
    val queryPoints = sparsePoints.sample(withReplacement = false, fraction = 0.1)
    val completeNN = fastModel.fullNeighbors(k).flatMap{ case (entry, neigh) =>
      neigh.map{ case (realDistance, e) =>
        val estDistance = EuclideanDistance.apply(entry.point.features, e.point.features)
        if(realDistance > 0)
          math.abs((realDistance - estDistance) / realDistance)
        else 
          .0f
      }  
    }.cache()
    val averageError = completeNN.sum() / completeNN.count()
    println("Average error distance: " + averageError)
    val fastnn = fastModel.neighbors(queryPoints, k).collectAsMap().toMap
    
    /** Apply batch nearest neighbor in local **/
    val localnn = batchNearestNeighbor(queryPoints.collect(), densePoints.collect(), k)
    compareResults(fastnn, localnn)
    
  }

  test("find neighbors for a set of query points") {
    val fastAnn =
      new fastANN(dimensions)
        .setRandomSeed(1234567)
    
    val fastModel = fastAnn.fastANNtrain(sparsePoints, nClasses)
    val queryPoints = sparsePoints.sample(withReplacement = false, fraction = 0.01)
    val fastnn = fastModel.neighbors(queryPoints, 10).collectAsMap().toMap
    
    assert(fastnn.size == queryPoints.count())
    
    runAssertions(fastnn.toArray)
  }
}

object fastANNSuite {

  def runAssertions(neighbors: Array[(Long, Array[(Long, Double)])]): Unit = {

    // At least some neighbors are found
    assert(neighbors.size > 0)

    neighbors.foreach {
      case (id1, distances) => {
        var maxDist = 0.0
        distances.foreach {
          case (id2, distance) => {
            // No neighbor pair contains the same ID twice
            assert(id1 != id2)

            // The neighbors are sorted in ascending order of distance
            assert(distance >= maxDist)
            maxDist = distance
          }
        }
      }
    }
  }
  
  def batchNearestNeighbor(query: Seq[IDPoint], 
      casebase: Seq[IDPoint], k: Int): Map[Long, Array[(Long, Double)]] ={
    
    val ordering = Ordering[Double].on[(Double, IDPoint)](_._1).reverse
    
    query.map{ q =>
      var topk = new BoundedPriorityQueue[(Double, IDPoint)](k)(ordering)
      casebase.map{ c =>
        val dist = EuclideanDistance.apply(q._2.features, c._2.features)
        topk += dist -> c
      }
      val partitions = topk.map{ case (_, idp) => math.floor(Vectors.norm(idp._2.features, 2) % 4) }
      println("Partitions: " + partitions)
      (q._1, topk.map{ case (dist, idp) => idp._1 -> dist}.toArray )
    }.toMap
  }
  
  
  def compareResults(aproxNeighbors: Map[Long, Array[(Long, Double)]],
                    exactNeighbors: Map[Long, Array[(Long, Double)]]): Unit = {
    
    val diff = aproxNeighbors.map{ case(key, neig1) =>
      val neig2 = exactNeighbors.getOrElse(key, Array.empty[(Long, Double)])
      math.abs(neig1.map(_._2).sum - neig2.map(_._2).sum)
    }
    
    val avgNumberNeig = exactNeighbors.map(_._2.size).sum / exactNeighbors.size

    val ndistinct = aproxNeighbors.map{ case(key, neig1) =>
      val asd = exactNeighbors.getOrElse(key, Array.empty[(Long, Double)])
      val s2 = exactNeighbors.getOrElse(key, Array.empty[(Long, Double)]).map(_._1).toSet
      val s1 = neig1.map(_._1).toSet
      s1.size - s1.intersect(s2).size
    }
    
    val mean = diff.sum / diff.size    
    val std = math.sqrt(diff.map(a => math.pow(a - mean, 2)).sum / (diff.size - 1))
    
    println("Average number of neighbors: " + avgNumberNeig)
    println("Average difference between neighbors (approximate): " + mean)
    println("Standard deviation for differences (approximate): " + std)
    println("Total number of distinct neighbors between versions: " + ndistinct.sum)
    println("Average distinct neighbors between versions: " + ndistinct.sum.toFloat / ndistinct.size)

  }
}
