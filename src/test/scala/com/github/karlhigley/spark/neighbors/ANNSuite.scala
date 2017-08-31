package com.github.karlhigley.spark.neighbors

import org.scalatest.FunSuite
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.{ Vector => MLLibVector }
import com.github.karlhigley.spark.neighbors.lsh.HashTableEntry
import org.apache.spark.ml.feature.LabeledPoint
import com.github.karlhigley.spark.neighbors.ANNModel.IDPoint

class ANNSuite extends FunSuite with TestSparkContext {
  import ANNSuite._

  val numPoints = 1000
  val dimensions = 100
  val density = 0.5

  var sparsePoints: RDD[IDPoint] = _
  var densePoints:  RDD[IDPoint] = _

  override def beforeAll() {
    super.beforeAll()
    val localPoints = TestHelpers.generateRandomPoints(numPoints, dimensions, density)
    sparsePoints = sc.parallelize(localPoints).zipWithIndex.map{ case(v, id) => (id, new LabeledPoint(-1, v))}
    densePoints = sparsePoints.mapValues(slp => new LabeledPoint(slp.label, slp.features.toDense))
  }

  test("compute hamming nearest neighbors as a batch") {
    val ann =
      new ANN(dimensions, "hamming")
        .setTables(1)
        .setSignatureLength(16)

    val model = ann.train(sparsePoints)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect()
    val localNeighbors = neighbors.collect()

    runAssertions(localHashTables, localNeighbors)
  }

  test("compute cosine nearest neighbors as a batch") {
    val ann =
      new ANN(dimensions, "cosine")
        .setTables(1)
        .setSignatureLength(4)

    val model = ann.train(sparsePoints)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect()
    val localNeighbors = neighbors.collect()

    runAssertions(localHashTables, localNeighbors)
  }

  test("compute euclidean nearest neighbors as a batch") {
    val ann =
      new ANN(dimensions, "euclidean")
        .setTables(1)
        .setSignatureLength(4)
        .setBucketWidth(2)

    val model = ann.train(sparsePoints)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect()
    val localNeighbors = neighbors.collect()

    runAssertions(localHashTables, localNeighbors)
  }

  test("compute euclidean nearest neighbors as a batch - dense vectors") {
    val ann =
      new ANN(dimensions, "euclidean")
        .setTables(1)
        .setSignatureLength(4)
        .setBucketWidth(2)

    val model = ann.train(densePoints)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect
    val localNeighbors  = neighbors.collect
    val nrNeighbors     = localNeighbors.length

    println(s"Euclidean nr neighbors: $nrNeighbors")

    runAssertions(localHashTables, localNeighbors)
  }

  test("compute manhattan nearest neighbors as a batch") {
    val ann =
      new ANN(dimensions, "manhattan")
        .setTables(1)
        .setSignatureLength(4)
        .setBucketWidth(25)

    val model = ann.train(sparsePoints)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect
    val localNeighbors  = neighbors.collect
    val nrNeighbors     = localNeighbors.length

    println(s"Manhattan nr neighbors: $nrNeighbors")

    runAssertions(localHashTables, localNeighbors)
  }

  test("compute fractional nearest neighbors as a batch") {
    val ann =
      new ANN(dimensions, "fractional")
        .setTables(10)
        .setSignatureLength(4)
        .setBucketWidth(250)

    val model = ann.train(sparsePoints)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect
    val localNeighbors  = neighbors.collect
    val nrNeighbors     = localNeighbors.length

    println(s"Manhattan nr neighbors: $nrNeighbors")

    runAssertions(localHashTables, localNeighbors)
  }

  test("compute jaccard nearest neighbors as a batch") {
    val ann =
      new ANN(dimensions, "jaccard")
        .setTables(1)
        .setSignatureLength(8)
        .setBands(4)
        .setPrimeModulus(739)

    val model = ann.train(sparsePoints)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect()
    val localNeighbors = neighbors.collect()

    runAssertions(localHashTables, localNeighbors)
  }

  test("with multiple hash tables neighbors don't contain duplicates") {
    val localPoints = TestHelpers.generateRandomPoints(numPoints, dimensions, density)
    val withDuplicates = localPoints ++ localPoints
    val points = sc.parallelize(withDuplicates).zipWithIndex.map{ case(v, id) => (id, new LabeledPoint(-1, v))}

    val ann =
      new ANN(dimensions, "hamming")
        .setTables(4)
        .setSignatureLength(16)

    val model = ann.train(points)
    val neighbors = model.neighbors(10)

    val localNeighbors = neighbors.collect()

    localNeighbors.foreach {
      case (id1, distances) => {
        val neighborSet = distances.map {
          case (id2, distance) => id2
        }.toSet

        assert(neighborSet.size == distances.size)
      }
    }
  }

  test("find neighbors for a set of query points") {
    val ann =
      new ANN(dimensions, "cosine")
        .setTables(1)
        .setSignatureLength(4)

    val model = ann.train(sparsePoints)

    val queryPoints = sparsePoints.sample(withReplacement = false, fraction = 0.01)
    val approxNeighbors = model.neighbors(queryPoints, 10)

    assert(approxNeighbors.count() == queryPoints.count())
  }
}

object ANNSuite {

  def runAssertions(hashTables: Array[_ <: HashTableEntry[_]],
                    neighbors: Array[(Long, Array[(Long, Double)])]): Unit = {

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
