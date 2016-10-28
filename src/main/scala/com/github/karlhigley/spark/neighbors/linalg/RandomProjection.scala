package com.github.karlhigley.spark.neighbors.linalg

import java.util.Random

import org.apache.commons.math3.distribution.CauchyDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY
import org.apache.commons.math3.distribution.{CauchyDistribution => ApacheCauchyDistribution}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Vector}

/**
 * A simple random projection based on Spark's existing
 * random generation and multiplication of dense matrices.
 */
class RandomProjection(val matrix: DenseMatrix) extends Serializable {

  /**
   * Apply the projection to supplied vector
   */
  def apply(vector: Vector): DenseVector = matrix.multiply(vector)

}

object RandomProjection {

  /**
   * Generate a random projection based on the input and output
   * dimensions
   */
  def generateGaussian(originalDim: Int, projectedDim: Int, random: Random): RandomProjection = {
    val localMatrix = DenseMatrix.randn(projectedDim, originalDim, random)
    new RandomProjection(localMatrix)
  }

  def generateCauchy(originalDim: Int, projectedDim: Int, random: Random): RandomProjection = {
    def bla(numRows: Int, numCols: Int): DenseMatrix = {
      checkInputs(numRows, numCols)

      val cauchyDistribution = new ApacheCauchyDistribution(0, 1, DEFAULT_INVERSE_ABSOLUTE_ACCURACY)
      new DenseMatrix(numRows, numCols, cauchyDistribution.sample(numRows * numCols))
    }

    val localMatrix = bla(projectedDim, originalDim)
    new RandomProjection(localMatrix)
  }

  private def checkInputs(numRows: Int, numCols: Int): Unit = {
    require(
      numRows.toLong * numCols <= Int.MaxValue,
      s"$numRows x $numCols dense matrix is too large to allocate"
    )
  }

}