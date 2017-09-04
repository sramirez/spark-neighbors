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

@RunWith(classOf[JUnitRunner])
class fastANNSuite extends FunSuite with TestSparkContext {
  import fastANNSuite._

  val numPoints = 1000
  val dimensions = 10
  val density = 0.75
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
    
    val fastModel = fastAnn.fastANNtrain(sparsePoints, nClasses)
    val queryPoints = sparsePoints.sample(withReplacement = false, fraction = 0.1)
    val completeNN = fastModel.fullNeighbors(k).flatMap{ case (entry, neigh) =>
      neigh.map{ case (realDistance, e) =>
        val estDistance = fastModel.approxEuclidDistance(entry, e) match {
          case Some(e1) => e1._1
          case None => .0f
        }
        (realDistance - estDistance) / realDistance
      }  
    }.cache()
    val averageError = completeNN.sum() / completeNN.count()
    val fastnn = fastModel.neighbors(queryPoints, k).collectAsMap().toMap
    
    /*val simpleAnn =
      new ANN(dimensions, "euclidean")
        .setTables(1)
        .setSignatureLength(4)
        .setBucketWidth(2)

    val simpleModel = simpleAnn.train(densePoints)
    val localnn = simpleModel.neighbors(queryPoints, 10).collectAsMap().toMap*/
    
    val localnn = batchNearestNeighbor(queryPoints.collect(), densePoints.collect(), k)

    //runAssertions(fastnn.toArray)
    compareResults(fastnn, localnn)
    
  }

  test("find neighbors for a set of query points") {
    val fastAnn =
      new fastANN(dimensions)
    
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
        if(c._1 != q._1){
          val dist = EuclideanDistance.apply(q._2.features, c._2.features)
          topk += dist -> c
        }
      }
      (q._1, topk.map{ case (dist, idp) => idp._1 -> dist}.toArray )
    }.toMap
  }
  
  
  def compareResults(aproxNeighbors: Map[Long, Array[(Long, Double)]],
                    exactNeighbors: Map[Long, Array[(Long, Double)]]): Unit = {

    // At least some neighbors are found
    //assert(aproxNeighbors.size > 0 && aproxNeighbors.size == exactNeighbors.size)
    
    val diff = aproxNeighbors.map{ case(key, neig1) =>
      val neig2 = exactNeighbors.getOrElse(key, Array.empty[(Long, Double)])
      neig1.map(_._2).sum - neig2.map(_._2).sum
    }
    
    val avgNumberNeig = exactNeighbors.map(_._2.size).sum / exactNeighbors.size
    println("Average number of neighbors: " + avgNumberNeig)
    val ndistinct = aproxNeighbors.map{ case(key, neig1) =>
      val asd = exactNeighbors.getOrElse(key, Array.empty[(Long, Double)])
      val s2 = exactNeighbors.getOrElse(key, Array.empty[(Long, Double)]).map(_._1).toSet
      val s1 = neig1.map(_._1).toSet
      s1.size - s1.intersect(s2).size
    }
    
    val mean = diff.sum / diff.size
    println("Average difference between neighbors (approximate): " + mean)
    val std = math.sqrt(diff.map(a => math.pow(a - mean, 2)).sum / (diff.size - 1))
    println("Standard deviation for differences (approximate): " + std)
    println("Total number of distinct neighbors between versions: " + ndistinct.sum)
    println("Average distinct neighbors between versions: " + ndistinct.sum.toFloat / ndistinct.size)
    
    //val limit = 100
    //assert(mean < limit)
  }
}
