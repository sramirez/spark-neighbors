package com.github.karlhigley.spark.neighbors.linalg

import java.util.Random

import breeze.stats.distributions.{CauchyDistribution, LevyDistribution}
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
    def randc(numRows: Int, numCols: Int): DenseMatrix = {
      checkInputs(numRows, numCols)

      val cauchyDistribution = new CauchyDistribution(0, 1)
      new DenseMatrix(numRows, numCols, cauchyDistribution.drawMany(numRows * numCols))
    }

    val localMatrix = randc(projectedDim, originalDim)
    new RandomProjection(localMatrix)
  }

  def generateLevy(originalDim: Int, projectedDim: Int, random: Random): RandomProjection = {
    def randl(numRows: Int, numCols: Int): DenseMatrix = {
      checkInputs(numRows, numCols)

      val levyDistribution = new LevyDistribution(0, 1)
      new DenseMatrix(numRows, numCols, levyDistribution.drawMany(numRows * numCols))
    }

    val localMatrix = randl(projectedDim, originalDim)
    new RandomProjection(localMatrix)
  }

  private def checkInputs(numRows: Int, numCols: Int): Unit = {
    require(
      numRows.toLong * numCols <= Int.MaxValue,
      s"$numRows x $numCols dense matrix is too large to allocate"
    )
  }

}