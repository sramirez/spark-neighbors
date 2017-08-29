package com.github.karlhigley.spark.neighbors.lsh

import org.apache.spark.ml.feature.LabeledPoint

/**
 * Abstract base class for locality-sensitive hash functions.
 */
abstract class LSHFunction[+T <: Signature[_]] {

  /**
   * Compute the hash signature of the supplied vector
   */
  def signature(v: LabeledPoint): T

  /**
   * Build a hash table entry for the supplied vector
   */
  def hashTableEntry(id: Long, table: Int, v: LabeledPoint): HashTableEntry[T]

}