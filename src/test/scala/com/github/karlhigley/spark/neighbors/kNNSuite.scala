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
import org.apache.spark.util.AccumulatorV2
import com.github.karlhigley.spark.neighbors.util.breezeMatrixAccumulator
import scala.collection.immutable.HashSet
import breeze.linalg.Axis
import breeze.linalg._
import breeze.numerics._

@RunWith(classOf[JUnitRunner])
class kNNSuite extends FunSuite with TestSparkContext {
  import kNNSuite._

  val numPoints = 1000
  val dimensions = 100
  val density = 0.5
  val nClasses = 3
  val k = 10

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
    
    val knn = new KNN()    
    val model = knn.train(densePoints)
    val queryPoints = sparsePoints.sample(withReplacement = false, fraction = 0.1)
    val neighbors = model.neighbors(queryPoints, k)
    
    val nf = densePoints.first()._2.features.size
    val accum = new breezeMatrixAccumulator(nf, nf)
    neighbors.mapPartitions{case it =>
      val matrix = breeze.linalg.DenseMatrix.zeros[Float](nf, nf)
      while(it.hasNext){
        val (elem, nn) = it.next()
        nn.foreach { neighbor =>  
          var set = new HashSet[Int]()
          elem.features.foreachActive{ (index, value) =>           
            if(neighbor.features(index) - value == 0)
              set += index
          }
          set.foreach(i => set.foreach(j => matrix(i, j) += 1))
        }        
      }
      accum.add(matrix)
      it
    }
    val a = accum.value.toDenseMatrix
    val collisions = breeze.linalg.sum(a(*, ::)).toArray.zipWithIndex.sortBy(_._1)
    print(collisions.mkString("\n")) 
  }
}

object kNNSuite {

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
}
